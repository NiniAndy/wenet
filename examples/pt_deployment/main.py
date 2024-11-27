from __future__ import print_function

import yaml
from argparse import Namespace

from examples.pt_deployment.subfunction import inference, test, train
from wenet.utils.config import override_config


if __name__ == '__main__':
    stage = "train"

    args_path = "/ssd/zhuang/code/wenet/examples/pt_deployment/args.yaml"
    with open(args_path, 'r', encoding='utf-8') as file:
        args = yaml.safe_load(file)

    args = Namespace(**args)


    if stage == "train":
        with open(args.config, 'r', encoding='utf-8') as file:
            configs = yaml.safe_load(file)
        if len(args.override_config) > 0:
            configs = override_config(configs, args.override_config)

        train(args, configs)

    elif stage == "test":
        with open(args["config"], 'r', encoding='utf-8') as file:
            configs = yaml.safe_load(file)

        # configs["ckpt"] = "/ssd/zhuang/code/FunASR/examples/ChineseCorrection/tb_logs/stage_1/version_0/checkpoints/model.pt.avg10"

        test(configs)

    elif stage == "inference":
        with open(args["config"], 'r', encoding='utf-8') as file:
            model_configs = yaml.safe_load(file)

        configs = {**model_configs, **args}
        # configs["ckpt"] = "/ssd/zhuang/code/FunASR/examples/ChineseCorrection/tb_logs/pny2han/version_0/checkpoints/epoch=88-val_loss=7.8599-val_asr_acc=0.9454.ckpt"

        test_data = "那她杀的阿姨在机场的照片"

        inference(test_data, configs)

    pass