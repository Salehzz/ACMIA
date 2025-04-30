import os
import torch
import logging
import argparse
import jsonlines
import pickle as pkl
from tqdm import tqdm
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch.nn.functional as F
import numpy as np

logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)

# Define the transformation function
def transform_example(example):
    # Prepare separate lists for texts and labels
    texts = []
    labels = []
    # Add member samples with label 1
    # Add nonmember samples with label 0
    for i in range(len(example['member'])):
        text = example['member'][i]
        texts.append(text)
        labels.append(1)
        text = example['nonmember'][i]
        texts.append(text)
        labels.append(0)
    return {'input': texts, 'label': labels}

# Flatten the lists of texts and labels into individual records
def flatten_batch(batch):
    return {
        'input': batch['input'],
        'label': batch['label']
    }

# helper functions
def convert_huggingface_data_to_list_dic(dataset):
    all_data = []
    for i in range(len(dataset)):
        ex = dataset[i]
        all_data.append(ex)
    return all_data


def get_arg():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--ref_dat", type=str, default='Chinese', 
    choices=[
        'English',
        'Chinese',
    ]
    )
    parser.add_argument(
    '--split', type=str, default='ngram_7_0.2', 
    choices=[
        'ngram_7_0.2',
        'ngram_13_0.2',
        'ngram_13_0.8'
    ]
    )
    parser.add_argument("--fil_num", type=int, default=1)
    parser.add_argument("--model", type=str, default="Qwen/Qwen1.5-72B")
    parser.add_argument("--max_tok", type=int, default=2048)
    parser.add_argument("--sav_int", type=int, default=1e6)
    parser.add_argument("--out_dir", type=str, default="Your output directory")

    arg = parser.parse_args()
    arg.dataset = arg.ref_dat

    return arg


def fre_dis(ref_dat, tok, ran_dis, max_tok, number_of_examples):
    """
    token frequency distribution
    ref_dat: reference dataset
    tok: tokenizer
    """
    for i, e in enumerate(tqdm(ref_dat, desc=f"Samples")):
        if(number_of_examples==i):
            break
        text = e["text"]
        input_ids = tok.encode(text)
        if(max_tok>0):
            input_ids = input_ids[:max_tok]
        values, counts = np.unique(input_ids, return_counts=True)
        ran_dis[values] += counts

    return ran_dis



def fre_dis_ch(ref_dat, tok, ran_dis, max_tok, k):
    """
    token frequency distribution
    ref_dat: reference dataset
    tok: tokenizer
    """
    for i, e in enumerate(tqdm(ref_dat, desc=f"{k+1} sub-dataset")):
        text = e["text"]
        input_ids = tok.encode(text)
        if(max_tok>0):
            input_ids = input_ids[:max_tok]
        values, counts = np.unique(input_ids, return_counts=True)
        ran_dis[values] += counts

    return ran_dis


if __name__ == "__main__":
    args = get_arg()

    out_dir = args.out_dir
    out_path = os.path.join(out_dir, args.ref_dat)
    Path(out_path).mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, trust_remote_code=True)

    ran_dis = [0] * model.config.vocab_size

    if args.ref_dat == "English":

        en = load_dataset("allenai/c4", "en" ,  streaming=True)['train']
        ran_dis = np.array(ran_dis)
        ran_dis = fre_dis(en, tokenizer, ran_dis, args.max_tok, args.sav_int)
        ran_dis = ran_dis.tolist()


    elif args.ref_dat == 'Chinese':
        ran_dis = np.array(ran_dis)
        for i in range(args.fil_num):
            iter = i
            while len(str(i)) < 4:
                i = "0" + str(i)
            fil_nam = f"part-{i}.jsonl"
            ref_dat_dir = 'ChineseWebText directory'
            ref_dat_pat = os.path.join(ref_dat_dir, fil_nam)
            with open(ref_dat_pat, "r+", encoding="utf8") as f:
                examples = []
                for example in tqdm(jsonlines.Reader(f)):
                    examples.append(example)

                if(args.fil_num==1):
                    ran_dis = fre_dis_ch(examples[:int(args.sav_int)], tokenizer, ran_dis, args.max_tok, iter)
                else:
                    ran_dis = fre_dis_ch(examples, tokenizer, ran_dis, args.max_tok, iter)

        ran_dis = ran_dis.tolist()


    with open(f"{out_path}/{args.model.split('/')[-1]}.pkl", "wb") as f:
        pkl.dump(ran_dis, f)