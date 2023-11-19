# -*- coding: utf-8 -*

import pandas as pd
from utils import *
from prompt_helper import *

from openai_helper import *

train_data_dir = "data/train_qa.json"
if os.path.exists(train_data_dir):
    with open(train_data_dir, 'r') as json_file:
        qa_dict = json.load(json_file)
    print("加载基础数据成功：{}".format(len(qa_dict)))
else:
    # 加载基础数据
    qa_dict = {}
    columns_to_convert = ['知识标题', '相似问法', '答案内容（网页）']
    for file_name in ['./data/SSE-2023-07-12-10_45_54.xlsx', './data/责任人-07-12-10_55_04.xlsx', './data/Expense-Receipt.xlsx']:
        df = pd.read_excel(file_name, engine='openpyxl')
        df[columns_to_convert] = df[columns_to_convert].astype(str)
        df.drop_duplicates(inplace=True)

        for index, row in df.iterrows():
            qa_dict[row['知识标题'].strip()] = row['答案内容（网页）'].strip()
            if row['相似问法'] is not None and row['相似问法'].strip() != "" and row['相似问法'].strip() != "nan":
                for q in row['相似问法'].split("\n"):
                    if q.strip() != "":
                        qa_dict[q.strip()] = row['答案内容（网页）'].strip()
    print("加载基础数据成功：{}".format(len(qa_dict)))  # 407行

    # GPT-4扩充训练数据
    failed_qs = list(qa_dict.keys())
    while failed_qs:
        k = failed_qs.pop(0)
        standalone_question_prompt = gen_standalone_question_prompt(k)
        try:
            gpt_res = parse_json(get_gpt_downgrade_res(standalone_question_prompt))
            standalone_questions = gpt_res["candidate_Standalone_Questions"]
            for sq in standalone_questions:
                qa_dict[sq] = qa_dict[k]
            print("origin q: {}, standalone qs: {}".format(k, standalone_questions))
        except:
            print("gen candidate_Standalone_Questions failed: {}".format(k))
            failed_qs.append(k)
    with open(train_data_dir, 'w') as f:
        f.write(json.dumps(qa_dict, ensure_ascii=False))
    print("GPT-4扩充训练数据成功：{}".format(len(qa_dict)))

# 构造finetune格式数据
qs = list(qa_dict.keys())
print("get_batch_emb started...")
emb_index = get_batch_emb(qs)
print("open ai get_batch_emb finished...")

ft_baichuan_data_dir = "data/ft_baichuan_data.json"
data_dic_list = []

for i, k in enumerate(qs):
    if i % 100 == 0:
        print("{} is processing...".format(i))

    ctxs, sims = gen_emb_context(emb_index[i], emb_index, qs, 6)
    pos_ctx, neg_ctx = process_ctxs(ctxs, k, qa_dict)

    # 正样本
    item_dic_pos = {"id": str(i) + "_1", "conversations": []}
    item_dic_pos["conversations"].append({
        "from": "human",
        "value": gen_retrieve_prompt(pos_ctx, "User: " + k)
    })
    item_dic_pos["conversations"].append({
        "from": "gpt",
        "value": qa_dict[k] + "\n"
    })
    data_dic_list.append(item_dic_pos)

    # 负样本
    if neg_ctx == "":
        continue
    item_dic_neg = {"id": str(i) + "_0", "conversations": []}
    item_dic_neg["conversations"].append({
        "from": "human",
        "value": gen_retrieve_prompt(neg_ctx, "User: " + k)
    })
    item_dic_neg["conversations"].append({
        "from": "gpt",
        "value": "对不起，基于现有的知识，暂时找不到该问题的正确答案。\n"
    })
    data_dic_list.append(item_dic_neg)
with open(ft_baichuan_data_dir, 'w') as f:
    f.write(json.dumps(data_dic_list, ensure_ascii=False))
print("All finished...")
