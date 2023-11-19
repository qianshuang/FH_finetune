# -*- coding: utf-8 -*

import json
import re

import numpy as np
from boltons.iterutils import remap


def remove_dic_null(dic):
    doc_temp = remap(dic, visit=lambda path, key, value: value is not None and value != "" and value != [])
    return doc_temp


def parse_json(text):
    try:
        first_brace_index = text.find("{")
        last_brace_index = text.rfind("}")
        text = text[first_brace_index:last_brace_index + 1]

        state = json.loads(text, strict=False)
        state = remove_dic_null(state)
    except:
        try:
            result = re.sub(r'[,:.](?=\s*})', '', text)  # 匹配逗号后面紧跟着的大括号，去掉该逗号
            state = json.loads(result, strict=False)
            state = remove_dic_null(state)
        except:
            print("can not parse json from: " + text)
            state = {}
    return state


def gen_emb_context(query_emb, faiss_index, qs, emb_sim_cand_num=4, emb_sim_threshold=0.5):
    vector_norm = np.linalg.norm(query_emb)
    batch_vectors_norm = np.linalg.norm(faiss_index, axis=1)
    scores = np.dot(faiss_index, query_emb) / (batch_vectors_norm * vector_norm)
    valid_indices = np.where(scores > emb_sim_threshold)[0]
    top_indices = np.argsort(scores[valid_indices])[::-1][:emb_sim_cand_num]
    top_valid_indices = valid_indices[top_indices]
    return np.array(qs)[top_valid_indices], scores[top_valid_indices]


def process_ctxs(ctxs, k, qa_dict):
    remain_qs_pos = []
    remain_qs_neg = []
    for ctx in ctxs:
        if ctx == k:
            continue
        if qa_dict[ctx] == qa_dict[k]:
            remain_qs_pos.append(ctx)
        else:
            remain_qs_neg.append(ctx)

    if len(remain_qs_pos) == 0 and len(remain_qs_neg) == 0:
        return "Question: " + k + "\n" + "Answer: " + qa_dict[k], ""
    elif len(remain_qs_pos) > 0 and len(remain_qs_neg) == 0:
        return "Question: " + remain_qs_pos[0] + "\n" + "Answer: " + qa_dict[remain_qs_pos[0]], ""
    elif len(remain_qs_pos) == 0 and len(remain_qs_neg) > 0:
        pos_ct = [k, remain_qs_neg[0]]
        pos_res = "\n------------\n".join(["Question: " + ctx + "\n" + "Answer: " + qa_dict[ctx] for ctx in pos_ct])
        neg_res = "Question: " + remain_qs_neg[0] + "\n" + "Answer: " + qa_dict[remain_qs_neg[0]]
        return pos_res, neg_res
    else:
        pos_ct = [remain_qs_pos[0], remain_qs_neg[0]]
        pos_res = "\n------------\n".join(["Question: " + ctx + "\n" + "Answer: " + qa_dict[ctx] for ctx in pos_ct])
        neg_res = "Question: " + remain_qs_neg[0] + "\n" + "Answer: " + qa_dict[remain_qs_neg[0]]
        return pos_res, neg_res
