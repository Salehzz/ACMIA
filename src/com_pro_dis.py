import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import json
import torch
import openai
import logging
import argparse
import pickle as pkl
from tqdm import tqdm
from pathlib import Path
from tensor_parallel import TensorParallelPreTrainedModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from datasets import load_dataset
import torch.nn.functional as F

logging.getLogger().setLevel(logging.INFO)

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
    parser.add_argument("--tar_mod", type=str, default="Qwen/Qwen1.5-72B")
    parser.add_argument(
    '--dataset', type=str, default='PatentMIA', 
    choices=[
        'WikiMIA_length32',
        'WikiMIA_length64',
        'WikiMIA_length128', 
        'WikiMIA_length256', 
        'WikiMIA_length32_paraphrased',
        'WikiMIA_length64_paraphrased',
        'WikiMIA_length128_paraphrased', 
        'arxiv',
        'dm_mathematics', 
        'github', 
        'hackernews', 
        'pile_cc', 
        'pubmed_central', 
        'wikipedia_(en)', 
        'PatentMIA',
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
    parser.add_argument("--max_tok", type=int, default=1024)
    parser.add_argument("--temp", type=int, default=25)
    parser.add_argument("--key_nam", type=str, default="input")
    parser.add_argument('--half', action='store_true', default=True)
    parser.add_argument('--int8', action='store_true', default=False)
    parser.add_argument('--int4', action='store_true', default=False)
    parser.add_argument("--out_dir", type=str, default="Your output directory")
    # ref model
    parser.add_argument('--ref', action='store_true', default=False)

    arg = parser.parse_args()
    return arg


# load model
def load_model(name,args):
    half_kwargs = {}
    if args.int4 or args.int8:
        if args.int4:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        elif args.int8:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        
        model = AutoModelForCausalLM.from_pretrained(
            name, return_dict=True, device_map='auto', quantization_config=quantization_config, trust_remote_code=True,
        )
    else:
        if args.half:
            half_kwargs = dict(torch_dtype=torch.bfloat16)

        model = AutoModelForCausalLM.from_pretrained(
            name, return_dict=True, device_map='auto', **half_kwargs, trust_remote_code=True,
        )

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    return model, tokenizer


def model_output(text, model, tok):

    device = model.device
    input_ids = (tok.encode(text,return_tensors='pt',truncation=True,max_length=args.max_tok)).to(device)

    with torch.no_grad():
        output = model(input_ids, labels=input_ids)

    loss = -output.loss.item()
    logits = output.logits
    logits.detach()

    return logits, input_ids, loss


def inference(text, label, tar_mod, tar_tok, Tempratures, is_ref):

    response = {}
    logits, input_ids, _ = model_output(text, tar_mod, tar_tok)

    logits = logits[0, :-1].to(dtype=torch.float32, copy=False)

    input_ids = input_ids[0][1:]
    input_ids_expanded = input_ids.unsqueeze(-1)
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=input_ids_expanded).squeeze(-1)

    response["input_ids"] = input_ids.tolist()
    response["log_probs"] = token_log_probs.tolist()
    
    response["text"] = text
    response["label"] = label



    if(not is_ref):

        _, _, lower_loss = model_output(text.lower(), tar_mod, tar_tok)
        response["lower_loss"] = lower_loss

        mu = (probs * log_probs).sum(-1)
        sigma = (probs * torch.square(log_probs)).sum(-1) - torch.square(mu)
        mink_plus = (token_log_probs-mu)/sigma.sqrt()

        response["mink_plus"] = mink_plus.tolist()


        log_probs_temprature = []
        mu_temprature = []
        sigma_temprature = []

        log_probs_expanded = log_probs.unsqueeze(0)

        split_num = len(Tempratures)
        number_temp = 0
        for k in range(split_num):

            if(k==split_num-1):
                split_tempratures = Tempratures[int(k*len(Tempratures)/split_num):]
            else:
                split_tempratures = Tempratures[int(k*len(Tempratures)/split_num):int((k+1)*len(Tempratures)/split_num)]


            new_log_probs = F.log_softmax(log_probs_expanded * split_tempratures, dim=-1)

            if(number_temp!=split_tempratures.size(0)):
                number_temp = split_tempratures.size(0)
                input_ids_expanded = input_ids.unsqueeze(0).expand(split_tempratures.size(0), -1).unsqueeze(-1)

            new_token_log_probs = new_log_probs.gather(dim=-1, index=input_ids_expanded).squeeze(-1)
            log_probs_temprature.extend(new_token_log_probs.tolist())


            mu = (new_log_probs.exp() * new_log_probs).sum(-1,keepdim=True)
            sigma = (new_log_probs.exp() * torch.square(new_log_probs-mu)).sum(-1).sqrt()

            mu_temprature.extend(mu.squeeze(-1).tolist())
            sigma_temprature.extend(sigma.tolist())


        response[f"log_probs_temprature"] = log_probs_temprature
        response[f"mu_temprature"] = mu_temprature
        response[f"sigma_temprature"] = sigma_temprature

    return response


def tok_pro_dis(dat, key_nam, tar_mod, tar_tok, max_temp, is_ref):

    # Constants
    Tempratures = [0.0]
    constant = 0.1

    for i in range(-max_temp,max_temp+1):
        Tempratures.append(2.0**(i*constant))


    Tempratures = torch.tensor(Tempratures, dtype=torch.float32, device=tar_mod.device).view(-1, 1, 1)

    responses = []
    for example in tqdm(dat):
        # print(example)
        text = example[key_nam]
        label = example["label"]
        responses.append(inference(text, label, tar_mod, tar_tok, Tempratures, is_ref))

    return  responses

if __name__ == '__main__':
    args = get_arg()
    logging.info(f"compute token probability distribution from {args.tar_mod} on {args.dataset}")

    out_dir = args.out_dir
    out_path = os.path.join(os.getcwd(),out_dir, args.dataset)

    if 'WikiMIA' not in args.dataset and 'patent' not in args.dataset.lower():
        out_path = os.path.join(out_path,args.split)

    Path(out_path).mkdir(parents=True, exist_ok=True)

    # load dataset
    if 'WikiMIA' in args.dataset:
        if not (('paraphrased' in args.dataset) or ('perturbed' in args.dataset)):
            dataset = load_dataset('swj0419/WikiMIA', split=args.dataset)
        else:
            dataset = load_dataset('zjysteven/WikiMIA_paraphrased_perturbed', split=args.dataset)

    elif 'patent' in args.dataset.lower():
        file_path = 'PatentMIA.jsonl Directory'
        dataset = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line)
                new_record = {'input': record['text'], 'label': record['label']}
                dataset.append(new_record)
        
    else:
        dataset = load_dataset("iamgroot42/mimir",args.dataset, token='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',trust_remote_code=True)[args.split]
        transformed_data = dataset.map(transform_example, batched=True, remove_columns=['member', 'nonmember', 'member_neighbors', 'nonmember_neighbors'])
        dataset = transformed_data.map(flatten_batch, batched=True)
    
    dataset = convert_huggingface_data_to_list_dic(dataset)


    tar_model, tar_tokenizer = load_model(args.tar_mod,args=args)
    pro_dis = tok_pro_dis(dataset, args.key_nam, tar_model, tar_tokenizer, args.temp, args.ref)
    
    model_name = args.tar_mod.split('/')[-1]
    with open(f"{out_path}/{model_name}.pkl", "wb") as f:
        pkl.dump(pro_dis, f)
