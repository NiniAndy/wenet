# Copyright (c) 2020 Mobvoi Inc (Binbin Zhang)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import datetime
import logging
import sys
from contextlib import nullcontext

# if your python version < 3.7 use the below one
# from contextlib import suppress as nullcontext
import torch

from wenet.utils.common import StepTimer
from wenet.utils.train_utils import (wenet_join,
                                     batch_forward,
                                     batch_backward,
                                     update_parameter_and_lr,
                                     log_per_step,
                                     save_checkpoint)
import os
from deepspeed.utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict
from wenet.utils.fsdp_utils import fsdp_save_model
import yaml


class Executor:

    def __init__(self,
                 global_step: int = 0,
                 device: torch.device = torch.device("cpu"),
                 monitor = "loss",
                 save_n =  10,
    ):
        self.monitor = monitor
        self.save_n = save_n
        if monitor == "acc":
            self.best = 0.0
        elif monitor == "loss":
            self.best = float("inf")
        else:
            raise ValueError(f"monitor: {monitor} not supported")
        self.monitor_list = []
        self.model_list = []

        self.step = global_step + 1
        self.train_step_timer = None
        self.cv_step_timer = None
        self.device = device

    def train(self, model, optimizer, scheduler, train_data_loader,
              cv_data_loader, writer, configs, scaler, group_join=None):
        ''' Train one epoch
        '''
        if self.train_step_timer is None:
            self.train_step_timer = StepTimer(self.step)
        model.train()
        info_dict = copy.deepcopy(configs)
        logging.info('using accumulate grad, new batch size is {} times larger than before'.format(info_dict['accum_grad']))
        # A context manager to be used in conjunction with an instance of
        # torch.nn.parallel.DistributedDataParallel to be able to train
        # with uneven inputs across participating processes.
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_context = model.join
        else:
            model_context = nullcontext

        with model_context():
            for batch_idx, batch_dict in enumerate(train_data_loader):
                info_dict["tag"] = "TRAIN"
                info_dict["step"] = self.step
                info_dict["batch_idx"] = batch_idx
                if group_join is not None and wenet_join(group_join, info_dict):
                    break

                if batch_dict["target_lengths"].size(0) == 0:
                    continue

                context = None
                # Disable gradient synchronizations across DDP processes.
                # Within this context, gradients will be accumulated on module
                # variables, which will later be synchronized.
                if info_dict.get("train_engine", "torch_ddp") in ["torch_ddp", "torch_fsdp"] and (batch_idx + 1) % info_dict["accum_grad"] != 0:
                    context = model.no_sync
                # Used for single gpu training and DDP gradient synchronization
                # processes.
                else:
                    context = nullcontext

                with context():
                    info_dict = batch_forward(model, batch_dict, scaler, info_dict, self.device)
                    info_dict = batch_backward(model, scaler, info_dict)

                info_dict = update_parameter_and_lr(model, optimizer, scheduler, scaler, info_dict)
                # write training: tensorboard && log
                log_per_step(writer, info_dict, timer=self.train_step_timer)
                save_interval = info_dict.get('save_interval', sys.maxsize)
                if (self.step +1) % save_interval == 0 and self.step != 0 and (batch_idx + 1) % info_dict["accum_grad"] == 0:
                    import torch.distributed as dist
                    # Ensure all ranks start CV at the same time in step mode
                    dist.barrier()
                    loss_dict = self.cv(model, cv_data_loader, configs)
                    model.train()
                    info_dict.update({
                        "tag":
                        "step_{}".format(self.step),
                        "loss_dict":
                        loss_dict,
                        "save_time":
                        datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S'),
                        "lrs":
                        [group['lr'] for group in optimizer.param_groups]
                    })
                    self.save_model(model, info_dict)
                    # write final cv: tensorboard
                    log_per_step(writer, info_dict)
                    # Ensure all ranks start Train at the same time in step mode
                    dist.barrier()
                self.step += 1 if (batch_idx + 1) % info_dict["accum_grad"] == 0 else 0

    def cv(self, model, cv_data_loader, configs):
        ''' Cross validation on '''
        if self.cv_step_timer is None:
            self.cv_step_timer = StepTimer(0.0)
        else:
            self.cv_step_timer.last_iteration = 0.0
        model.eval()
        info_dict = copy.deepcopy(configs)
        num_seen_utts, loss_dict, total_acc = 1, {}, []  # avoid division by 0
        with torch.no_grad():
            for batch_idx, batch_dict in enumerate(cv_data_loader):
                info_dict["tag"] = "CV"
                info_dict["step"] = self.step
                info_dict["batch_idx"] = batch_idx
                info_dict["cv_step"] = batch_idx

                num_utts = batch_dict["target_lengths"].size(0)
                if num_utts == 0:
                    continue

                info_dict = batch_forward(model, batch_dict, None, info_dict, self.device)
                _dict = info_dict["loss_dict"]

                num_seen_utts += num_utts
                total_acc.append(_dict['th_accuracy'].item() if _dict.get('th_accuracy', None) is not None else 0.0)
                for loss_name, loss_value in _dict.items():
                    if loss_value is not None and "loss" in loss_name and torch.isfinite(loss_value):
                        loss_value = loss_value.item()
                        loss_dict[loss_name] = loss_dict.get(loss_name, 0) + loss_value * num_utts
                # write cv: log
                log_per_step(writer=None, info_dict=info_dict, timer=self.cv_step_timer)
        for loss_name, loss_value in loss_dict.items():
            loss_dict[loss_name] = loss_dict[loss_name] / num_seen_utts
        loss_dict["acc"] = sum(total_acc) / len(total_acc)
        return loss_dict


    def save_model(self, model, info_dict):
        rank = int(os.environ.get('RANK', 0))
        tag = info_dict["tag"]
        model_dir = info_dict["model_dir"]
        save_model_path = os.path.join(model_dir, '{}.pt'.format(tag))
        # save ckpt
        if info_dict["train_engine"] == "deepspeed":
            # NOTE(xcsong): All ranks should call this API, but only rank 0 save the general model params. see:
            # https://github.com/microsoft/DeepSpeed/issues/2993
            with torch.no_grad():
                model.save_checkpoint(save_dir=model_dir, tag=tag, client_state=info_dict)
                if info_dict["save_states"] == "model_only" and rank == 0:
                    convert_zero_checkpoint_to_fp32_state_dict(model_dir, save_model_path, tag=tag)
                    os.system("rm -rf {}/{}".format(model_dir, tag))

        elif info_dict['train_engine'] == "torch_fsdp":
            fsdp_save_model(model, save_model_path, info_dict)

        elif rank == 0:
            save_checkpoint(model, save_model_path, info_dict)

        # save yaml
        if rank == 0:
            with open("{}/{}.yaml".format(model_dir, tag), 'w') as fout:
                data = yaml.dump(info_dict)
                fout.write(data)

        if rank == 0:
            loss_dict = info_dict.get("loss_dict", None)
            # 初始化时没有loss_dict
            if loss_dict is None:
                if self.monitor == "acc":
                    self.monitor_list.append(0.0)
                elif self.monitor == "loss":
                    self.monitor_list.append(float("inf"))
                else:
                    raise ValueError(f"monitor: {self.monitor} not supported")
                self.model_list.append(save_model_path)

            # 如果有loss_dict时先将monitor_index加入monitor_list然后排序，将最差的删除
            else:
                monitor_index = loss_dict[self.monitor]
                self.monitor_list.append(monitor_index)
                self.model_list.append(save_model_path)

                if self.monitor == "acc":
                    sorted_indices = sorted(range(len(self.monitor_list)), key=lambda k: self.monitor_list[k], reverse=True) # 从大到小排序
                elif self.monitor == "loss":
                    sorted_indices = sorted(range(len(self.monitor_list)), key=lambda k: self.monitor_list[k])  # 从小到大排序
                else:
                    raise ValueError(f"monitor: {self.monitor} not supported")

                # Sort both monitor_list and model_list based on sorted indices
                self.monitor_list = [self.monitor_list[i] for i in sorted_indices]
                self.model_list = [self.model_list[i] for i in sorted_indices]

                if len(self.monitor_list) > self.save_n:
                    # 删除最差的model
                    del_monitor_index = self.monitor_list.pop()
                    del_model_path = self.model_list.pop()
                    del_model_path = os.path.join(os.getcwd(), del_model_path)
                    os.remove(del_model_path)
                    del_model_yaml = del_model_path.replace(".pt", ".yaml")
                    os.remove(del_model_yaml)
                    logging.info(f"[Rank {rank}] Delete model:  {del_model_path} with {self.monitor} {del_monitor_index}")

                logging.info(f"[Rank {rank}] Best model:    {self.model_list[0]} with {self.monitor} {self.monitor_list[0]}")
                # 确保model_list和monitor_list长度一致
                assert len(self.monitor_list) == len(self.model_list), "monitor_list and model_list length not equal"
                # 确保model_list和monitor_list长度不超过save_n
                assert len(self.monitor_list) <= self.save_n, "monitor_list and model_list length exceed save_n"



