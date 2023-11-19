# -*- coding:utf-8 -*-

import json
import os

os.environ['OPENAI_API_BASE'] = 'https://oaapi.dianxiaomi.com/v1'
import openai
import requests

openai.api_key = "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"


def get_emb(query_arr):
    response = openai.Embedding.create(
        input=query_arr, engine="text-embedding-ada-002"
    )
    return [d["embedding"] for d in response["data"]]


def call_chatgpt(prompt):
    response = openai.ChatCompletion.create(
        # model="gpt-3.5-turbo",
        model="gpt-4",
        temperature=0,
        messages=prompt
    )
    return response.choices[0].message["content"]


def call_azure(prompt, api_url="https://comm100gpt.openai.azure.com/openai/deployments/Summary/chat/completions?api-version=2023-03-15-preview", key="3a4cc360c1f8427c9a22225197744697"):
    query_body = {"temperature": 0, "messages": [{"role": "user", "content": prompt}]}
    headers = {'content-type': 'application/json', 'api-key': key}
    response = requests.post(api_url, json=query_body, headers=headers, timeout=60)
    json_res = json.loads(response.text)
    return json_res["choices"][0]["message"]["content"]


# print(get_emb(["aaa", "aaa", "aaa"]))

p_ = """nihao"""
# print(call_azure(p_))
print(call_chatgpt([{"role": "user", "content": p_}]))
