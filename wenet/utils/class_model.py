#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright [2023-11-28] <sxc19@mails.tsinghua.edu.cn, Xingchen Song>

from wenet.LLM.causallm_model import CausalLM
from wenet.ctl_model.asr_model_ctl import CTLModel
from wenet.k2.model import K2Model
from wenet.paraformer.paraformer import Paraformer
from wenet.transducer.transducer import Transducer
from wenet.transformer.asr_model import ASRModel
from wenet.whisper.whisper import Whisper
from wenet.paraformer.paraformer_V2 import ParaformerV2
from wenet.cn_correction.text_bert import TextBert
from wenet.text_speech_bimodal_asr.text_speech_asr import TextSpeechASR



WENET_MODEL_CLASSES = {
    "asr_model": ASRModel,
    "ctl_model": CTLModel,
    "whisper": Whisper,
    "k2_model": K2Model,
    "transducer": Transducer,
    'paraformer': Paraformer,
    'paraformerV2': ParaformerV2,
    'causal_llm': CausalLM,
    'text_bert': TextBert,
    'text_speech_asr': TextSpeechASR
}