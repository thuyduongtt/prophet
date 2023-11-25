# ------------------------------------------------------------------------------ #
# Author: Zhenwei Shao (https://github.com/ParadoxZW)
# Description: Runner that handles the prompting process
# ------------------------------------------------------------------------------ #

import os, sys
# sys.path.append(os.getcwd())

import pickle
import json, time
import math
import random
import argparse
from datetime import datetime
from copy import deepcopy
import yaml
from pathlib import Path
# import openai
import requests

from .utils.fancy_pbar import progress, info_column
from .utils.data_utils import Qid2Data
from configs.task_cfgs import Cfgs

from llama import Llama, Dialog
from typing import List, Optional

llama_generator = None


class Runner:
    def __init__(self, __C, evaluater):
        self.__C = __C
        self.evaluater = evaluater
        # openai.api_key = __C.OPENAI_KEY

        # print('OpenAI Key:', os.getenv("AZURE_OPENAI_KEY"))
        # openai.api_key = os.getenv("AZURE_OPENAI_KEY")
        # openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
        # openai.api_type = 'azure'
        # openai.api_version = '2023-05-15'

    def gpt3_infer(self, prompt_text, _retry=0):
        # print(prompt_text)
        # exponential backoff
        # if _retry > 0:
        #     print('retrying...')
        #     st = 2 ** _retry
        #     # time.sleep(st)

        # if self.__C.DEBUG:
        #     # print(prompt_text)
        #     # time.sleep(0.05)
        #     return 0, 0

        # print('calling gpt3...')
        # response = openai.Completion.create(
        #     engine=self.__C.MODEL,
        #     prompt=prompt_text,
        #     temperature=self.__C.TEMPERATURE,
        #     max_tokens=self.__C.MAX_TOKENS,
        #     logprobs=1,
        #     stop=["\n", "<|endoftext|>"],
        #     # timeout=20,
        # )

        # use internal OpenAI server
        # response = openai.Completion.create(
        #     engine='gpt35',
        #     prompt=prompt_text,
        #     temperature=self.__C.TEMPERATURE,
        #     max_tokens=self.__C.MAX_TOKENS,
        #     # logprobs=1,  # not supported in GPT 3.5
        #     stop=["\n", "<|endoftext|>"],
        # )

        # LLAMA API
        # API_KEY = "f9392cca-fcac-4fc1-9126-ffa767da8649"
        # API_BASE = "https://ews-emea.api.bosch.com/knowledge/insight-and-analytics/llms/d/v1"
        # MODEL = "meta-llama/Llama-2-13b-chat-hf"
        #
        # headers = {
        #     "api-key": API_KEY,
        #     "Content-Type": "application/json"
        # }
        #
        # body = {
        #     "model": MODEL,
        #     "prompt": prompt_text,
        #     "temperature": self.__C.TEMPERATURE,
        #     "max_tokens": self.__C.MAX_TOKENS,
        #     "logprobs": 1,
        #     "stop": ["\n", "<|endoftext|>"]
        # }
        # response = requests.post(API_BASE + '/completions', data=json.dumps(body), headers=headers)
        # response = response.json()
        # response_txt = response.choices[0].text.strip()
        # print(response_txt, 'len of tokens:', len(response['tokens']))

        # LLAMA LOCAL
        global llama_generator
        if llama_generator is None:
            init_llama(self.__C.LLAMA_MODEL, self.__C.LLAMA_TOKENIZER)

        prompts: List[str] = [prompt_text]

        print(prompt_text)

        response = llama_generator.text_completion(
            prompts,
            temperature=self.__C.TEMPERATURE,
            max_gen_len=self.__C.MAX_TOKENS,
            logprobs=True
        )[0]

        # print('Response')
        # print(response)
        # print('End Response')

        response_txt = response['generation']  # Llama-2
        print(response_txt, 'len of tokens:', len(response['tokens']))

        plist = []
        for ii in range(len(response['tokens'])):
            if response['tokens'][ii] in ["\n", "<|endoftext|>"]:
                break
            plist.append(response['logprobs'][ii])
        prob = math.exp(sum(plist))

        return response_txt, prob

    def sample_make(self, ques, capt, cands, ans=None):
        line_prefix = self.__C.LINE_PREFIX
        cands = cands[:self.__C.K_CANDIDATES]
        prompt_text = line_prefix + f'Context: {capt}\n'
        prompt_text += line_prefix + f'Question: {ques}\n'
        cands_with_conf = [f'{cand["answer"]}({cand["confidence"]:.2f})' for cand in cands]
        cands = ', '.join(cands_with_conf)
        prompt_text += line_prefix + f'Candidates: {cands}\n'
        prompt_text += line_prefix + 'Answer:'
        if ans is not None:
            prompt_text += f' {ans}'
        return prompt_text

    def get_context(self, example_qids):
        # making context text for one testing input
        prompt_text = self.__C.PROMPT_HEAD
        examples = []
        for key in example_qids:
            ques = self.trainset.get_question(key)
            caption = self.trainset.get_caption(key)
            cands = self.trainset.get_topk_candidates(key)
            gt_ans = self.trainset.get_most_answer(key)
            examples.append((ques, caption, cands, gt_ans))
            prompt_text += self.sample_make(ques, caption, cands, ans=gt_ans)
            prompt_text += '\n\n'
        return prompt_text

    def run(self):
        ## where logs will be saved
        Path(self.__C.LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
        with open(self.__C.LOG_PATH, 'w') as f:
            f.write(str(self.__C) + '\n')
        ## where results will be saved
        Path(self.__C.RESULT_DIR).mkdir(parents=True, exist_ok=True)

        self.cache = {}
        self.cache_file_path = os.path.join(
            self.__C.RESULT_DIR,
            'cache.json'
        )
        if self.__C.RESUME:
            self.cache = json.load(open(self.cache_file_path, 'r'))

        print(
            'Note that the accuracies printed before final evaluation (the last printed one) are rough, just for checking if the process is normal!!!\n')
        self.trainset = Qid2Data(
            self.__C,
            self.__C.TRAIN_SPLITS,
            True
        )
        self.valset = Qid2Data(
            self.__C,
            self.__C.EVAL_SPLITS,
            self.__C.EVAL_NOW,
            json.load(open(self.__C.EXAMPLES_PATH, 'r'))
        )

        # if 'aok' in self.__C.TASK:
        #     from evaluation.aokvqa_evaluate import AOKEvaluater as Evaluater
        # else:
        #     from evaluation.okvqa_evaluate import OKEvaluater as Evaluater
        # evaluater = Evaluater(
        #     self.valset.annotation_path,
        #     self.valset.question_path
        # )

        infer_times = self.__C.T_INFER
        N_inctx = self.__C.N_EXAMPLES

        count = 0
        for qid in progress.track(self.valset.qid_to_data, description="Working...  "):
            if qid in self.cache:
                continue

            count += 1
            if count % 1000 == 0:
                print(f'{count} / {len(self.valset.qid_to_data)}')

            ques = self.valset.get_question(qid)
            caption = self.valset.get_caption(qid)
            cands = self.valset.get_topk_candidates(qid, self.__C.K_CANDIDATES)

            prompt_query = self.sample_make(ques, caption, cands)
            example_qids = self.valset.get_similar_qids(qid, k=infer_times * N_inctx)
            random.shuffle(example_qids)

            prompt_info_list = []
            ans_pool = {}
            # multi-times infer
            for t in range(infer_times):
                # print(f'Infer {t}...')
                prompt_in_ctx = self.get_context(example_qids[(N_inctx * t):(N_inctx * t + N_inctx)])
                prompt_text = prompt_in_ctx + prompt_query
                gen_text, gen_prob = self.gpt3_infer(prompt_text)

                ans = self.evaluater.prep_ans(gen_text)
                if ans != '':
                    ans_pool[ans] = ans_pool.get(ans, 0.) + gen_prob

                prompt_info = {
                    'prompt': prompt_text,
                    'answer': gen_text,
                    'confidence': gen_prob
                }
                prompt_info_list.append(prompt_info)
                # time.sleep(self.__C.SLEEP_PER_INFER)

            # vote
            if len(ans_pool) == 0:
                answer = self.valset.get_topk_candidates(qid, 1)[0]['answer']
            else:
                answer = sorted(ans_pool.items(), key=lambda x: x[1], reverse=True)[0][0]

            self.evaluater.add(qid, answer)
            self.cache[qid] = {
                'question_id': qid,
                'answer': answer,
                'prompt_info': prompt_info_list
            }
            json.dump(self.cache, open(self.cache_file_path, 'w'))

            ll = len(self.cache)
            if self.__C.EVAL_NOW and not self.__C.DEBUG:
                if ll > 21 and ll % 10 == 0:
                    rt_accuracy = self.valset.rt_evaluate(self.cache.values())
                    info_column.info = f'Acc: {rt_accuracy}'

        self.evaluater.save(self.__C.RESULT_PATH)
        if self.__C.EVAL_NOW:
            with open(self.__C.LOG_PATH, 'a+') as logfile:
                self.evaluater.evaluate(logfile)


def init_llama(model_path, tokenizer_path):
    print(f'Init Llama model ({model_path}, {tokenizer_path})')
    global llama_generator
    llama_generator = Llama.build(
        ckpt_dir=model_path,
        tokenizer_path=tokenizer_path,
        max_seq_len=4096,
        max_batch_size=1,
    )
    print('Model initialized.')


def prompt_login_args(parser):
    parser.add_argument('--debug', dest='DEBUG', help='debug mode', action='store_true')
    parser.add_argument('--resume', dest='RESUME', help='resume previous run', action='store_true')
    parser.add_argument('--task', dest='TASK', help='task name, e.g., ok, aok_val, aok_test', type=str, required=True)
    parser.add_argument('--version', dest='VERSION', help='version name', type=str, required=True)
    parser.add_argument('--cfg', dest='cfg_file', help='optional config file', type=str, default='configs/prompt.yml')
    parser.add_argument('--examples_path', dest='EXAMPLES_PATH',
                        help='answer-aware example file path, default: "assets/answer_aware_examples_for_ok.json"',
                        type=str, default=None)
    parser.add_argument('--candidates_path', dest='CANDIDATES_PATH',
                        help='candidates file path, default: "assets/candidates_for_ok.json"', type=str, default=None)
    parser.add_argument('--captions_path', dest='CAPTIONS_PATH',
                        help='captions file path, default: "assets/captions_for_ok.json"', type=str, default=None)
    # parser.add_argument('--openai_key', dest='OPENAI_KEY', help='openai api key', type=str, default=None)
    parser.add_argument('--llama_model', dest='LLAMA_MODEL', help='', type=str, default=None)
    parser.add_argument('--llama_tokenizer', dest='LLAMA_TOKENIZER', help='', type=str, default=None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Heuristics-enhanced Prompting')
    prompt_login_args(parser)
    args = parser.parse_args()
    __C = Cfgs(args)
    with open(args.cfg_file, 'r') as f:
        yaml_dict = yaml.load(f, Loader=yaml.FullLoader)
    __C.override_from_dict(yaml_dict)
    print(__C)

    runner = Runner(__C)
    runner.run()
