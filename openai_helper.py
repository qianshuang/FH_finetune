# -*- coding: utf-8 -*-

import json
import os
import traceback

os.environ['OPENAI_API_BASE'] = 'https://oaapi.dianxiaomi.com/v1'
import time

import numpy as np
import requests

import openai
from openai.error import RateLimitError, APIError

ak = "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"


def get_emb(query_arr):
    # 最大输入长度8191
    num_retries = 10
    response = None

    for _ in range(num_retries):
        try:
            openai.api_key = ak
            response = openai.Embedding.create(
                input=query_arr, engine="text-embedding-ada-002"
            )
            embs = [d["embedding"] for d in response["data"]]
            return embs
        except Exception:
            sleep_time = 0.5
            ("{} call openai embedding api failed, sleep {} second...".format(ak, sleep_time))
            time.sleep(sleep_time)
    if response is None:
        raise RuntimeError("Failed to get openai embedding response...")


def get_batch_emb(query_arr):
    results = []
    # 分批调用
    sub_arrays = np.array_split(query_arr, 1 if len(query_arr) <= 1000 else (len(query_arr) / 1000 + 1))
    for arr in sub_arrays:
        try:
            sub_embs = get_emb(arr.tolist())
            results.extend(sub_embs)
        except:
            raise
    try:
        print("get_batch_emb shape: {}".format(np.array(results).shape))
    except:
        for i, item in enumerate(results):
            if item is None or len(item) != 1536:
                emb_res = get_emb([query_arr[i]])[0]  # 重试
                if emb_res is None or len(emb_res) != 1536:
                    traceback.print_exc()
                else:
                    results[i] = emb_res
    return results


def call_azure(prompt_messages, api_url, key):
    query_body = {"temperature": 0, "messages": prompt_messages}
    headers = {'content-type': 'application/json', 'api-key': key}
    response = requests.post(api_url, json=query_body, headers=headers, timeout=300)
    json_res = json.loads(response.text)
    try:
        return json_res["choices"][0]["message"]["content"]
    except:
        print("call_azure failed, prompt: {}, returned json result: {}".format(prompt_messages[1]["content"], json_res))
        raise RuntimeError("Failed to call_azure...")


def get_chatgpt_res(prompt_messages):
    try:
        endpoint = "https://comm100gpttest.openai.azure.com/openai/deployments/GPTChat/chat/completions?api-version=2023-03-15-preview"
        key = "e5fda94edc774de7b4fa75a663a7cd5a"
        return call_azure(prompt_messages, endpoint, key)
    except:
        traceback.print_exc()

    num_retries = 10
    response = None

    for _ in range(num_retries):
        try:
            openai.api_key = ak
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-16k",
                temperature=0,
                messages=prompt_messages
            )
            return response.choices[0].message["content"]
        except (RateLimitError, APIError):
            sleep_time = 0.5
            print("call chatgpt failed, sleep {} second...".format(sleep_time))
            time.sleep(sleep_time)
        except:
            traceback.print_exc()
            raise RuntimeError("Failed to get_chatgpt_res...")
    if response is None:
        raise RuntimeError("Failed to get_chatgpt_res...")


def get_gpt4_res(prompt_messages):
    num_retries = 5
    response = None

    for _ in range(num_retries):
        try:
            openai.api_key = ak
            response = openai.ChatCompletion.create(
                model="gpt-4",
                temperature=0,
                messages=prompt_messages
            )
            return response.choices[0].message["content"]
        except (RateLimitError, APIError):
            sleep_time = 0.5
            print("call gpt4 failed, sleep {} second...".format(sleep_time))
            time.sleep(sleep_time)
        except:
            traceback.print_exc()
            raise RuntimeError("Failed to get_gpt4_res...")
    if response is None:
        raise RuntimeError("Failed to get_gpt4_res...")


def get_gpt_downgrade_res(prompt):
    try:
        chat_res = get_gpt4_res(prompt)
    except:
        print("get_gpt4_res failed, trying chatgpt...")
        chat_res = get_chatgpt_res(prompt)
    return chat_res
