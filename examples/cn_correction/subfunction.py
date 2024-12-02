import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from yacs.config import CfgNode as CN

from examples.cn_correction.data import MyDataModule
from examples.cn_correction.lit_model import LitModel


class CustomProgressBar(TQDMProgressBar):
    def __init__(self):
        super(CustomProgressBar, self).__init__()

    def get_metrics(self, trainer, model):
        # don't show the version number
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)
        return items


def inference(inference_data, configs):
    data_module = MyDataModule(configs)
    model = LitModel(configs)
    # model.load_model(configs["ckpt"])
    output = model(inference_data)
    print (output)



def train(args, configs):

    data_module = MyDataModule(args, configs)
    configs =  data_module.configs
    model = LitModel(args, configs)

    bar = CustomProgressBar()

    save_conf = args.save_conf
    solver_conf = args.solver_conf
    save_file = save_conf.get("save_dir", "tb_logs")
    output_dir = getattr(args, "output_dir", "/ssd/zhuang/code/wenet/examples/pt_deployment")
    save_path = os.path.join(output_dir, save_file)

    checkpoint_callback = ModelCheckpoint(filename='{epoch}-{val_loss:.4f}-{val_asr_acc:.4f}',
                                          monitor='val_loss',
                                          save_top_k=save_conf.get("save_top_k", 10),
                                          mode='min',
                                          every_n_epochs=save_conf.get("every_n_epochs", 1)
                                          )

    trainer = pl.Trainer(devices=solver_conf.get("devices", [0]),
                         accelerator=solver_conf.get("accelerator", "gpu"),
                         accumulate_grad_batches=solver_conf.get("accumulate_grad_batches", 1),
                         strategy=solver_conf.get("strategy", "ddp"),
                         max_epochs=solver_conf.get("max_epochs", 100),
                         callbacks=[checkpoint_callback, bar],
                         logger=TensorBoardLogger(save_dir=save_path, name=save_conf.get("name", "default"))
                         )

    trainer.fit(model, data_module)


def test(configs):

    data_module = MyDataModule(configs)
    model = LitModel(configs)

    bar = CustomProgressBar()

    save_conf = configs["save_conf"]
    solver_conf = configs["solver_conf"]
    save_file = save_conf.get("save_dir", "tb_logs")
    output_dir = configs.get("output_dir", "/ssd/zhuang/code/FunASR/demo/")
    save_path = os.path.join(output_dir, save_file)

    trainer = pl.Trainer(devices=solver_conf.get("devices", [0]),
                         accelerator=solver_conf.get("accelerator", "gpu"),
                         logger=False,
                         callbacks=[bar])

    # model.load_model(configs["ckpt"])
    result = trainer.test(model, data_module)[0]

