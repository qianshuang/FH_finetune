# -*- coding: utf-8 -*

import json

with open("/Users/wcy/workspace/FH_finetune/data/ft_baichuan_data.json", 'r') as json_file:
    qa_dict = json.load(json_file)

res = []
for item in qa_dict:
    item["conversations"][0]["value"] = item["conversations"][0]["value"][0]["content"]
    res.append(item)

with open("/Users/wcy/workspace/FH_finetune/ft_baichuan_data.json", 'w') as f:
    f.write(json.dumps(res, ensure_ascii=False))
