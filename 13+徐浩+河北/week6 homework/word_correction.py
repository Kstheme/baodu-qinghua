import json
import copy
from ngram_language_model import NgramLanguageModel
import re
"""
文本纠错demo
加载同音字字典
加载语言模型
基本原理：
对于文本中每一个字，判断在其同音字中是否有其他字，在替换掉该字时，能使得语言模型计算的成句概率提高
"""


class Corrector:
    def __init__(self, language_model):
        #语言模型
        self.language_model = language_model
        #候选字字典
        self.sub_dict = self.load_tongyinzi("tongyin.txt")
        #成句概率的提升超过阈值则保留修改
        self.threshold = 7

    #实际上不光是同音字，同形字等也可以加入，本质上是常用的错字
    def load_tongyinzi(self, path):
        tongyinzi_dict = {}
        with open(path, encoding="utf8") as f:
            for line in f:
                char, tongyin_chars = line.split()
                tongyinzi_dict[char] = list(tongyin_chars)
        return tongyinzi_dict

    #纠错逻辑
    def correction(self, string):
        # print(self.sub_dict)
        fix_s = []
        yuan_prob = self.language_model.predict(string)
        max_prob = yuan_prob
        t_string = string
        for key, value in self.sub_dict.items():
            t_prob = self.language_model.predict(t_string)
            if re.findall(key, string) !=[]:
                for v in value:
                    tmp_string = re.sub(key, v, t_string)
                    tmp_prob = self.language_model.predict(tmp_string)
                    if tmp_prob > max_prob:
                        t_string = tmp_string
                        max_prob = tmp_prob
                    if tmp_prob >= 7:
                        fix_s.append(tmp_string)

        fix_s.append(t_string)
        print(self.language_model.predict(t_string))
        return fix_s


corpus = open("财经.txt", encoding="utf8").readlines()
lm = NgramLanguageModel(corpus, 3)

cr = Corrector(lm)
string = "每国货币政册空间不大"  #美国货币政策空间不大
string1 = "美国货币政策空间不大"
print(cr.load_tongyinzi("tongyin.txt"))
fix_string = cr.correction(string)

print("修改前：", string)
# print(cr.correction(string))
# print(lm.predict(string1))
print("修改后：", fix_string)