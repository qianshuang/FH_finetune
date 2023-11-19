# -*- coding: utf-8 -*


def gen_standalone_question_prompt(question):
    prompt_str = """GOAL:
Given the Question below, rephrase the Question to 5 candidate Standalone Questions with full intent, each candidate Standalone question must be able to fully express the meaning of the given Question, but in different ways.
Each candidate Standalone question must be in Chinese.
Focusing on key and potentially important information.
Let's work this out in a step by step way to be sure we have the right answer.

Question:
{}

You should only respond in JSON format as described below:
{{
    "reasoning": "reasoning",
    "candidate_Standalone_Questions": [5 candidate Standalone Questions in Chinese]
}}
Ensure the response can be parsed by Python json.loads""".format(question)
    return [{"role": "system", "content": "You are a trustworthy AI assistant."},
            {"role": "user", "content": prompt_str}]


def gen_retrieve_prompt(qa_retrieve_ctx, transed_dialogue):
    prompt_str = """目标:
参考下面的问答对知识（每个问答对由“------------”分隔），根据下面的对话生成最合适的答案。
一定要保证答案的完整性，尽可能使用对话和问答对知识中的原文作答，不要遗漏任何内容。
让我们一步一步地解决这个问题，以确保我们得到正确的答案。

问答对知识:
{}

对话:
{}

最终的答案是。\n""".format(qa_retrieve_ctx, transed_dialogue)
    return prompt_str
