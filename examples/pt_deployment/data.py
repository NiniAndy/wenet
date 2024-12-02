from abc import ABC

import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch

from wenet.utils.init_tokenizer import init_tokenizer
from wenet.utils.train_utils import init_dataset_and_dataloader, check_modify_and_save_config




class MyDataModule(pl.LightningDataModule):
    def __init__(self, args, configs: dict):
        """
        完成数据初始化，如定义数据集路径，需要的数据集名称等
        """
        super(MyDataModule, self).__init__()
        # 获取预训练模型名字
        tokenizer = init_tokenizer(configs)
        configs = check_modify_and_save_config(args, configs, tokenizer)

        self.tokenizer_dict = {"tokenizer": tokenizer}
        self.configs = configs
        self.args = args


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
            self.train_dataset, self.valid_dataset, self.dataloader_train, self.dataloader_valid = init_dataset_and_dataloader(
                self.args, self.configs, self.tokenizer_dict)
            pass
        if stage == 'test' or stage is None:
            self.test_dataset, self.valid_dataset, self.dataloader_test, self.dataloader_valid = init_dataset_and_dataloader(
                self.args, self.configs, self.tokenizer_dict)
            pass

    def train_dataloader(self):
        return self.dataloader_train

    def val_dataloader(self):
        return self.dataloader_valid

    def test_dataloader(self):
        return self.dataloader_test
