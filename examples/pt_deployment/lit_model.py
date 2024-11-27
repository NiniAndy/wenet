import pytorch_lightning as pl
import torch
import torch.optim as optim
from torch import nn
import os
import json
import logging
from argparse import Namespace

from wenet.utils.init_model import init_model
from wenet.bin.model_summary import model_summary
from wenet.utils.init_tokenizer import init_tokenizer
from wenet.utils.train_utils import  init_optimizer_and_scheduler




class LitModel(pl.LightningModule):
    """pytorch lightning 模型"""

    def __init__(self, args: Namespace, configs: dict):
        super(LitModel, self).__init__()
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.val_asr_acc = []
        self.val_loss = []

        model, configs = init_model(args, configs)

        torch.set_float32_matmul_precision(args.solver_conf.get("precision", "medium"))
        train_args = vars(args)
        train_configs = {**configs, **train_args}
        self.save_hyperparameters(train_configs)

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if local_rank == 0:
            logging.info(f"{model_summary(model)}")

        model, optimizer, scheduler = init_optimizer_and_scheduler(args, configs, model)

        self.model = model
        self.optim = optimizer
        self.scheduler = scheduler
        self.configs = configs


    def share_step(self, batch, batch_idx):
        result_dict = self.model(batch, "cuda")
        return result_dict

    def training_step(self, batch, batch_idx):
        result_dict = self.share_step(batch, batch_idx)
        loss = result_dict['loss']
        asr_acc = result_dict['th_accuracy']
        self.log("loss", loss, prog_bar=True, logger=True, sync_dist=True)
        self.log("asr_acc", asr_acc, prog_bar=True, logger=True, sync_dist=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        result_dict = self.share_step(batch, batch_idx)
        loss = result_dict['loss']
        asr_acc = result_dict['th_accuracy']
        self.val_loss.append(loss)
        self.val_asr_acc.append(asr_acc)
        # self.log("val_loss", loss, prog_bar=True, logger=True, sync_dist=True)
        # return {"val_loss": loss}

    def on_validation_epoch_end(self):
        avg_loss = 0
        avg_asr_acc = 0
        for i in range(len(self.val_loss)):
            avg_asr_acc += self.val_asr_acc[i]
            avg_loss += self.val_loss[i]
        avg_loss = avg_loss / len(self.val_loss)
        avg_asr_acc = avg_asr_acc / len(self.val_asr_acc)
        self.print('dev: loss: {:.4f}, acc: {:.4f} '.format(avg_loss, avg_asr_acc))
        self.log("val_loss", avg_loss, prog_bar=False, logger=True, sync_dist=True)
        self.log("val_asr_acc", avg_asr_acc, prog_bar=False, logger=True, sync_dist=True)
        return {"val_loss": avg_loss, "val_asr_acc": avg_asr_acc}

    def test_step(self, batch, batch_idx):
        self.step += 1
        inputs, input_len, tgt, tgt_len = batch
        hyps, _ = self.model.ctc_prefix_beam_search(
            inputs,
            input_len,
            beam_size=10,
            decoding_chunk_size=-1,
            num_decoding_left_chunks=-1,
            simulate_streaming=False,
            context_graph=None)
        hyps = list(hyps)
        cer, sub_dict = self.calculate_performance.calculate_cer(hyps, tgt[0].tolist(), self.config.eos_id)
        # self.print('测试集结果: cer: {:.2f}%'.format(cer * 100))
        self.log("test_cer", cer, on_epoch=True, prog_bar=True)
        return cer

    def forward(self, audio_path, inference_data):
        """
        推理部分
        Args:
            inference_data: 音频信号
            audio_path: 音频文件
            para_args: 载入参数
        Returns:
        """
        pass

    def configure_optimizers(self):
        return {
            "optimizer": self.optim,
            "lr_scheduler": self.scheduler  # 可选
        }

    def optimizer_step(self, epoch=None, batch_idx=None, optimizer=None, optimizer_closure=None):
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()
        lr = self.scheduler.get_lr()[0]
        self.log("lr", lr, prog_bar=False, logger=True, sync_dist=True)
        self.lr_schedulers().step()  # 这会调用 self.scheduler.step()
        # step = self.trainer.global_step + 1
        # rate = self.scheduler.get_lr()[0]
        # for p in optimizer.param_groups:
        #     p['lr'] = rate
        # self.log("lr", rate, prog_bar=False, logger=True, sync_dist=True)
        # optimizer.step(closure=optimizer_closure)

    def load_model(self, path):
        # ckpt = torch.load(path, map_location='cpu')
        # self.load_state_dict(ckpt["state_dict"])
        ckpt = torch.load(path, map_location='cpu')
        self.model.load_state_dict(ckpt)
