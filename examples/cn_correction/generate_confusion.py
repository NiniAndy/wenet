import json
from pypinyin import lazy_pinyin, INITIALS, FINALS_TONE3
from random import sample
import random
import numpy as np


class GenerateConfusion:
    """
    生成拼音混淆数据
    """

    def __init__(self, config):
        self.max_mask_prob = config["max_mask_prob"]
        self.mask_lv1_prob = config["mask_lv1_prob"]
        self.mask_lv2_prob = config["mask_lv2_prob"]
        self.mask_lv3_prob = config["mask_lv3_prob"]
        self.mask_lv4_prob = config["mask_lv4_prob"]
        self.mask_lv5_prob = config["mask_lv5_prob"]
        self.pinyin_set_path = config["pinyin_set_path"]
        self.initials_distence_path = config["initials_distence_path"]
        self.vowel_distence_path = config["vowel_distence_path"]

        assert self.mask_lv5_prob + self.mask_lv4_prob + self.mask_lv3_prob + self.mask_lv2_prob + self.mask_lv1_prob == 1.0

        self.level3_dict = {"zh": "z", "ch": "c", "sh": "s", "z": "zh", "c": "ch", "s": "sh",
                            "an": "ang", "en": "eng", "in": "ing", "ang": "an", "eng": "en", "ing": "in"}

        with open(self.pinyin_set_path, "r", encoding="utf-8") as f:
            self.pinyin_set = json.load(f)

        with open(self.initials_distence_path, "r", encoding="utf-8") as f:
            self.initials_distance_matrix = json.load(f)

        with open(self.vowel_distence_path, "r", encoding="utf-8") as f:
            self.vowel_distance_matrix = json.load(f)

    def check_pinyin(self, pinyin):
        """
        检查拼音是否有效
        """
        if pinyin in self.pinyin_set:
            if self.pinyin_set[pinyin]["total_num"] <= 10:
                return False
            else:
                return True
        else:
            return False

    def generate(self, sentence):
        """
        生成混淆数据
        """
        sentence_len = len(sentence)

        origin_initials = lazy_pinyin(sentence, INITIALS)  # 声母
        origin_vowel = lazy_pinyin(sentence, FINALS_TONE3)  # 韵母和音调
        origin_pinyin = [origin_initials[j] + origin_vowel[j] for j in range(len(origin_initials))]  # 完整拼音

        # assert len(origin_pinyin) == sentence_len
        mask_prob = np.random.uniform(0, self.max_mask_prob)
        sentence_list = [i for i in sentence]
        choice_num = int(sentence_len * mask_prob)
        sentence_index = [i for i in range(len(sentence))]
        word_choice_index = sample(sentence_index, choice_num)

        select_level_p = np.array(
            [self.mask_lv1_prob, self.mask_lv2_prob, self.mask_lv3_prob, self.mask_lv4_prob, self.mask_lv5_prob])
        for i in word_choice_index:
            # 选取混淆等级
            level = np.random.choice([1, 2, 3, 4, 5], p=select_level_p)
            # 记录原始文字，发音(完整拼音，声母，韵母，音调)
            origin_word = sentence_list[i]
            origin_word_pinyin = origin_pinyin[i]
            origin_word_initials = origin_initials[i]
            origin_word_vowel = origin_vowel[i]
            origin_word_tone = origin_word_vowel[-1] if origin_word_vowel[-1] in ["1", "2", "3", "4"] else ""

            change_flag = True

            if origin_word_tone != "":
                origin_word_vowel = origin_word_vowel[:-1]

            if level == 1:
                # 保持发音, 同音字
                if len(self.pinyin_set[origin_word_pinyin]["word_list"]) == 1:
                    level = 2
                else:
                    new_word_pinyin = origin_word_pinyin
            if level == 3:
                # 前后鼻音, 平舌音和卷舌音

                if origin_word_initials in self.level3_dict or origin_word_vowel in self.level3_dict:
                    if origin_word_initials in self.level3_dict and origin_word_vowel in self.level3_dict:
                        if random.random() < 0.5:
                            new_word_initials = self.level3_dict[origin_word_initials]
                            new_word_pinyin = new_word_initials + origin_word_vowel + origin_word_tone
                        else:
                            new_word_vowel = self.level3_dict[origin_word_vowel]
                            new_word_pinyin = origin_word_initials + new_word_vowel + origin_word_tone
                    elif origin_word_initials in self.level3_dict:
                        new_word_initials = self.level3_dict[origin_word_initials]
                        new_word_pinyin = new_word_initials + origin_word_vowel + origin_word_tone

                    else:
                        new_word_vowel = self.level3_dict[origin_word_vowel]
                        new_word_pinyin = origin_word_initials + new_word_vowel + origin_word_tone

                    # 有效性检查 有些拼音无对应的文字或者对应的字很少
                    if self.check_pinyin(new_word_pinyin):
                        pass
                        # print("check pass " + new_word_pinyin)
                    else:
                        level = 2
                        # print("check fail " + new_word_pinyin)
                        # print("Unable to generate confusion level will be set to 4")

                else:
                    # new_word_pinyin = origin_word_pinyin
                    level = 2
                    # print("no initials or vowel in confusion level will be set to 4")

            if level == 2:
                # 更换拼音音调
                check_flag = True
                tone_candidate_list = {"", "1", "2", "3", "4"}
                if origin_word_tone != "":
                    tone_candidate_list = tone_candidate_list - set(origin_word_tone)
                else:
                    tone_candidate_list = {"1", "2", "3", "4"}

                while check_flag:
                    confusion_tone = sample(list(tone_candidate_list), 1)
                    if origin_word_tone != "":
                        new_word_pinyin = origin_word_pinyin[:-1] + confusion_tone[0]
                    else:
                        new_word_pinyin = origin_word_pinyin + confusion_tone[0]
                    # 有效性检查 有些拼音无对应的文字或者对应的字很少
                    if self.check_pinyin(new_word_pinyin):
                        check_flag = False
                    else:
                        check_flag = True
                        tone_candidate_list = tone_candidate_list - set(confusion_tone)
                        if len(tone_candidate_list) == 0:
                            check_flag = False
                            level = 4

            if level == 4:
                # 拼音编辑距离为1，这里包含一部分 level3、2 无法生成有效拼音的情况
                if random.random() < 0.5:
                    # 修改辅音
                    try:
                        candidate_list = self.initials_distance_matrix[origin_word_initials]["1"]
                    except KeyError:
                        change_flag = False
                    else:
                        new_word_initials = sample(candidate_list, 1)[0]
                        new_word_pinyin = new_word_initials + origin_word_vowel + origin_word_tone
                else:
                    # 修改元音
                    try:
                        candidate_list = self.vowel_distance_matrix[origin_word_vowel]["1"]
                    except KeyError:
                        change_flag = False
                    else:
                        new_word_vowel = sample(candidate_list, 1)[0]
                        new_word_pinyin = origin_word_initials + new_word_vowel + origin_word_tone

                if self.check_pinyin(new_word_pinyin):
                    pass
                else:
                    change_flag = False

            if level == 5:
                if random.random() < 0.5:
                    # 修改辅音
                    try:
                        candidate_list = self.initials_distance_matrix[origin_word_initials]["other"]
                    except KeyError:
                        change_flag = False
                    else:
                        if len(candidate_list) == 0:
                            candidate_list = self.initials_distance_matrix[origin_word_initials]["1"]
                        new_word_initials = sample(candidate_list, 1)[0]
                        new_word_pinyin = new_word_initials + origin_word_vowel + origin_word_tone
                else:
                    # 修改元音
                    try:
                        candidate_list = self.vowel_distance_matrix[origin_word_vowel]["other"]
                    except KeyError:
                        change_flag = False
                    else:
                        new_word_vowel = sample(candidate_list, 1)[0]
                        new_word_pinyin = origin_word_initials + new_word_vowel + origin_word_tone

                if self.check_pinyin(new_word_pinyin):
                    pass
                    # print("check pass " + new_word_pinyin)
                else:
                    change_flag = False

            if change_flag:
                new_word_candidate_list = self.pinyin_set[new_word_pinyin]["word_list"].copy()
                p_select = self.pinyin_set[new_word_pinyin]["freq_list"].copy()
                if level == 1:
                    try:
                        origin_word_index = new_word_candidate_list.index(origin_word)
                    except ValueError:
                        pass
                    else:
                        new_word_candidate_list.pop(origin_word_index)
                        p_select.pop(origin_word_index)
                        p_new = [i * self.pinyin_set[new_word_pinyin]["total_num"] for i in p_select]
                        p_select = [i / sum(p_new) for i in p_new]

                p_select = np.array(p_select)
                new_word = np.random.choice(new_word_candidate_list, p=p_select)
                # print("level {}: {} changed to {}".format(level, origin_word, new_word))
                sentence_list[i] = new_word
            else:
                sentence_list[i] = origin_word
            pass
        new_sentence = "".join(sentence_list)
        assert len(sentence) == len(new_sentence)
        wrong_list = []
        for i in range(len(sentence)):
            if sentence[i] != new_sentence[i]:
                wrong_list.append(i)

        wrong_text = "".join(sentence_list)
        wrong_ids = wrong_list

        return wrong_text, wrong_ids


# class GenerateDelete:
#     """生成删除数据"""
#     def __init__(self, config):
#         self.config = config
#
#     def generate(self, wrong_text, wrong_ids):
#         """
#         生成混淆数据
#         """
#         sentence_len = len(sentence)
#
#         origin_initials = lazy_pinyin(sentence, INITIALS)  # 声母
#         origin_vowel = lazy_pinyin(sentence, FINALS_TONE3)  # 韵母和音调
#         origin_pinyin = [origin_initials[j] + origin_vowel[j] for j in range(len(origin_initials))]  # 完整拼音
#
#
# generate_confusion = GenerateConfusion(config.CONFUSION)
# wrong_text, wrong_ids = generate_confusion.generate("金华市环城小学校训没有一个字和学习有关却让学生家长和老师都掉下泪来")


