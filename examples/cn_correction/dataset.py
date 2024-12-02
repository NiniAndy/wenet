from torch.utils.data import Dataset
import torch
from examples.cn_correction.generate_confusion import GenerateConfusion
from functools import partial
import json
import numpy as np
from tqdm import tqdm
import os
from itertools import chain

from pypinyin import lazy_pinyin, INITIALS, FINALS_TONE3

def make_pny(sentence):
    origin_initials = lazy_pinyin(sentence, INITIALS)  # 声母
    origin_vowel = lazy_pinyin(sentence, FINALS_TONE3)  # 韵母和音调
    origin_pinyin = [origin_initials[j] + origin_vowel[j] for j in range(len(origin_initials))]
    return origin_pinyin



class ConfusionDataSet(Dataset):
    def __init__(self, tokenizer, pny_tokenizer, correct_path, confusion_config, mode):
        super(ConfusionDataSet, self).__init__()
        self.int_pad_value = -1
        self.float_pad_value = 0.0

        self.tokenizer = tokenizer
        self.pny_tokenizer  = pny_tokenizer
        self.generate = GenerateConfusion(confusion_config)
        self.correct_path = correct_path
        self.mode = mode

        if int(os.environ.get('LOCAL_RANK', 0)) == 0:
            print("加载", mode, "数据集：", correct_path)

        buffer = 1024 * 1024
        with open(correct_path) as f:
            length = sum(x.count('\n') for x in iter(partial(f.read, buffer), ''))

        self.correct_data = np.empty(length, dtype="<U64")
        # self.wrong_data = np.empty(length, dtype="<U64")
        # self.wrong_ids_data = np.empty(length, dtype="<U32")

        with open(correct_path, "r", encoding="utf-8") as f:
            i = 0
            for line in tqdm(f, total=length, desc="Processing lines"):
                if line != "\n" and len(line[:-1]) < 64:
                    self.correct_data[i] = line[:-1]
                    i += 1
        self.correct_data = self.correct_data[:i]


        # proc = ("/ssd/zhuang/code/FunASR/examples/aishell/conformer/exp/baseline_conformer_12e_6d_2048_256_zh_char_zhanwei/inference-model.pt.avg10"
        #      "/test_wenet_decoder/1best_recog/text.proc")
        # ref = ("/ssd/zhuang/code/FunASR/examples/aishell/conformer/exp/baseline_conformer_12e_6d_2048_256_zh_char_zhanwei/inference-model.pt.avg10"
        #      "/test_wenet_decoder/1best_recog/text.ref")
        # with open(proc, "r", encoding="utf-8") as f:
        #     self.proc_data = f.readlines()
        #
        # with open(ref, "r", encoding="utf-8") as f:
        #     self.ref_data = f.readlines()

    def __len__(self):
        return self.correct_data.size

    def __getitem__(self, item):
        correct_text = self.correct_data[item]

        if len(correct_text.split()) >1:
            correct_text = correct_text.split()[1 :]
            correct_text = "".join(correct_text)

        # if self.mode == "train":
        #     try:
        #         wrong_text, _ = self.generate.generate(correct_text)
        #     except:
        #         wrong_text = correct_text
        # else:
        #     wrong_text = correct_text

        # correct_text = self.ref_data[item]
        # wrong_text = self.proc_data[item]
        #
        # if len(correct_text.split()) > 1:
        #     correct_text = correct_text.split()[1:]
        #     correct_text = "".join(correct_text)
        #
        # if len(wrong_text.split()) > 1:
        #     wrong_text = wrong_text.split()[1:]
        #     wrong_text = "".join(wrong_text)


        try:
            wrong_text, _ = self.generate.generate(correct_text)
        except:
            wrong_text = correct_text


        correct_text_tokens, ids = self.tokenizer.tokenize(correct_text)
        text = torch.tensor(ids, dtype=torch.int64)

        wrong_pny_list = make_pny(wrong_text)
        blank = ['<blank>']
        wrong_pny_list = list(chain(*zip(wrong_pny_list, blank*(len(wrong_pny_list)-1)), [wrong_pny_list[-1]]))
        wrong_pny = self.pny_tokenizer.tokens2ids(wrong_pny_list)

        wrong_pny = torch.tensor(wrong_pny, dtype=torch.int64)
        pny_lengths =  torch.tensor([wrong_pny.size(0)], dtype=torch.int32)
        text_lengths =  torch.tensor([text.size(0)], dtype=torch.int32)

        sample = {
            "pny": wrong_pny,
            "pny_lengths": pny_lengths,
            "text": text,
            "text_lengths": text_lengths,
        }

        return sample





class ContrastDataSet_test(Dataset):
    def __init__(self, tokenizer, correct_path, error_path, mode):
        super(ContrastDataSet_test, self).__init__()
        self.tokenizer = tokenizer
        self.correct_path = correct_path
        self.error_path = error_path

        if int(os.environ.get('LOCAL_RANK', 0)) == 0:
            print("加载", mode, "数据集：", correct_path)

        buffer = 1024 * 1024

        with open(correct_path) as f:
            length = sum(x.count('\n') for x in iter(partial(f.read, buffer), ''))
        self.correct_data = np.empty(length, dtype="<U64")
        with open(correct_path, "r", encoding="utf-8") as f:
            i = 0
            for line in tqdm(f, total=length, desc="Processing lines"):
                if line != "\n" and len(line[:-1]) < 64:
                    self.correct_data[i] = line[:-1]
                    i += 1
        self.correct_data = self.correct_data[:i]

        with open(error_path, "r", encoding="utf-8") as f:
            length = sum(x.count('\n') for x in iter(partial(f.read, buffer), ''))
        self.error_data = np.empty(length, dtype="<U64")
        with open(error_path, "r", encoding="utf-8") as f:
            i = 0
            for line in tqdm(f, total=length, desc="Processing lines"):
                if line != "\n" and len(line[:-1]) < 64:
                    self.error_data[i] = line[:-1]
                    i += 1
        self.error_data = self.error_data[:i]

    def __len__(self):
        return self.correct_data.size

    def __getitem__(self, item):
        correct_text = self.correct_data[item]
        wrong_text = self.error_data[item]
        wrong_list = [0 if char1 == char2 else 1 for char1, char2 in zip(correct_text, wrong_text)]

        correct_text_tokens, correct_text_ids = self.tokenizer.tokenize(correct_text)
        wrong_text_tokens, wrong_text_ids = self.tokenizer.tokenize(wrong_text)
        correct_text_ids = torch.tensor(correct_text_ids).unsqueeze(0)
        wrong_text_ids = torch.tensor(wrong_text_ids).unsqueeze(0)

        wrong_list = torch.tensor(wrong_list).type_as(correct_text_ids)

        return wrong_text_ids, wrong_list, correct_text_ids
