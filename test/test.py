# -*- coding: utf-8 -*

import json
import random

data_train = json.load(open("../data/ft_baichuan_data.json"))
print("data_train length: {}".format(len(data_train)))

data_ori = json.load(open("../data/belle_chat_ramdon_10k.json"))
print("data_ori length: {}".format(len(data_ori)))

data_merge = data_train + data_ori[:5000]
print("data_merge length: {}".format(len(data_merge)))

random.shuffle(data_merge)
print(data_merge[:5])
