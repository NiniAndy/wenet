from abc import ABC

import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch

from wenet.utils.init_tokenizer import init_tokenizer, init_pny_tokenizer
from wenet.utils.train_utils import init_dataset_and_dataloader, check_modify_and_save_config
from examples.cn_correction.dataset import ConfusionDataSet


def collator(samples: list = None):
    outputs = {}
    for sample in samples:
        for key in sample.keys():
            if key not in outputs:
                outputs[key] = []
            outputs[key].append(sample[key])

    for key, data_list in outputs.items():
        if isinstance(data_list[0], torch.Tensor):
            if data_list[0].dtype == torch.int64 or data_list[0].dtype == torch.int32:
                pad_value = -1
            else:
                pad_value = 0.0

            outputs[key] = torch.nn.utils.rnn.pad_sequence(data_list, batch_first=True, padding_value=pad_value)
    return outputs




class MyDataModule(pl.LightningDataModule):
    def __init__(self, args, configs: dict):
        """
        完成数据初始化，如定义数据集路径，需要的数据集名称等
        """
        super(MyDataModule, self).__init__()
        # 获取预训练模型名字
        tokenizer = init_tokenizer(configs)
        pny_tokenizer = init_pny_tokenizer(configs)
        tokenizer_info = {"pny_tokenizer": pny_tokenizer, "tokenizer": tokenizer}
        configs = check_modify_and_save_config(args, configs, **tokenizer_info)

        self.tokenizer = tokenizer
        self.pny_tokenizer = pny_tokenizer
        self.configs = configs
        self.args = args

        self.confusion_conf = configs["dataset_conf"]["confusion_conf"]

        self.train_path = args.train_data
        self.valid_path = args.cv_data
        self.test_path = args.test_data

        self.train_batch_size = args.solver_conf["train_batch_size"]
        self.valid_batch_size = args.solver_conf["valid_batch_size"]
        self.test_batch_size = args.solver_conf["test_batch_size"]


    def prepare_data(self):
        """
        如果数据集需要下载可以在这里定义方法
        """
        pass

    def setup(self, stage=None):
        """
        数据集建立方法，框架自动调用
        :param stage: 当前阶段 fit/test
        """
        if stage == 'fit' or stage is None:
            self.train_dataset = ConfusionDataSet(self.tokenizer, self.pny_tokenizer, self.train_path, self.confusion_conf, mode='train')
            self.valid_dataset = ConfusionDataSet(self.tokenizer, self.pny_tokenizer, self.valid_path, self.confusion_conf, mode='valid')
            pass
        if stage == 'test' or stage is None:
            self.valid_dataset = ConfusionDataSet(self.tokenizer, self.pny_tokenizer, self.valid_path, self.confusion_conf, mode='valid')
            self.test_dataset = ConfusionDataSet(self.tokenizer, self.pny_tokenizer, self.test_path, self.confusion_conf, mode='test')
            pass

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, num_workers=12, collate_fn=collator)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.valid_batch_size, num_workers=12, collate_fn=collator)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.test_batch_size, num_workers=12, collate_fn=collator)
