# Copyright (c) 2022 Binbin Zhang (binbzha@qq.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from pprint import pprint

from wenet.LLM.causallm_model import CausalLM
from wenet.LLM.decoder import DecoderOnly

from wenet.finetune.lora.utils import (inject_lora_to_model, mark_only_lora_as_trainable)
from wenet.utils.checkpoint import load_checkpoint, load_trained_modules
from wenet.utils.class_model import WENET_MODEL_CLASSES



# def init_speech_model(args, configs):
#
#     model_class = configs.get('model', 'asr_model')
#     model = WENET_MODEL_CLASSES[model_class](**configs)
#
#     # if model_type in WENET_SSL_MODEL_CLASS.keys():
#     #     from wenet.ssl.init_model import init_model as init_ssl_model
#     #     model = init_ssl_model(configs, audio_encoder)
#
#     return model, configs
#
#
# def init_causal_llm(configs):
#     vocab_size = configs['output_dim']
#     assert configs['context_decoder'] == 'decoder_only'
#     assert configs['model'] == 'causal_lm'
#     decoder_only = DecoderOnly(**configs['decoder_conf'])
#
#     model = CausalLM(
#         vocab_size,
#         decoder_only,
#         **configs['model_conf'],
#         special_tokens=configs.get('tokenizer_conf', {}).get('special_tokens', None),
#     )
#     return model, configs





def init_model(args, configs):

    model_class = configs.get('model', 'asr_model')
    # configs['model'] = model_type
    # if model_type == 'causal_lm':
    #     model, configs = init_causal_llm(configs)
    # else:
    #     model, configs = init_speech_model(args, configs)
    configs['model'] = model_class
    if model_class == 'causal_lm':
        decoder_only = DecoderOnly(**configs['decoder_conf'])
        configs["decoder_only"] = decoder_only

    model = WENET_MODEL_CLASSES[model_class](**configs)

    if hasattr(args, 'use_lora') and args.use_lora:
        inject_lora_to_model(model, configs['lora_conf'])

    # If specify checkpoint, load some info from checkpoint
    if hasattr(args, 'checkpoint') and args.checkpoint is not None:
        infos = load_checkpoint(model, args.checkpoint)
    elif hasattr(args, 'enc_init') and args.enc_init is not None:
        infos = load_trained_modules(model, args)
    else:
        infos = {}
    configs["init_infos"] = infos

    if hasattr(args, 'use_lora') and args.use_lora:
        if hasattr(args, 'lora_ckpt_path') and args.lora_ckpt_path:
            load_checkpoint(model, args.lora_ckpt_path)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    freeze_param = configs.get("freeze_param", None)
    if freeze_param is not None:
        if "," in freeze_param:
            freeze_param = eval(freeze_param)
        if not isinstance(freeze_param, (list, tuple)):
            freeze_param = (freeze_param,)
        if local_rank == 0:
            print ("freeze_param is not None: %s", freeze_param)
        for t in freeze_param:
            for k, p in model.named_parameters():
                if k.startswith(t + ".") or k == t:
                    if local_rank == 0:
                        print (f"Setting {k}.requires_grad = False")
                    p.requires_grad = False

    if local_rank == 0:
        print("Model args:")
        pprint(vars(args))
        print("\n")
        print ("Model configs:")
        pprint(configs)
        print("\n")

    # Trye to tie some weights
    if hasattr(model, 'tie_or_clone_weights'):
        if not hasattr(args, 'jit'):
            args.jit = True  # i.e. export onnx/jit/ipex
        model.tie_or_clone_weights(args.jit)

    if hasattr(args, 'only_optimize_lora') and args.only_optimize_lora:
        mark_only_lora_as_trainable(model, bias='lora_only')

    return model, configs
