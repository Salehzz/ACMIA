import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,0"

import os, argparse
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, accuracy_score, average_precision_score
from tqdm import tqdm
import zlib
import gc
import jieba

import pickle as pkl

import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from datasets import load_dataset
import random
from transformers import set_seed
from copy import deepcopy

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

# arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="Qwen/Qwen1.5-32B")
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
        'PatentMIA'
    ]
)

parser.add_argument("--max_char", type=int, default=1e5)

parser.add_argument(
    '--split', type=str, default='ngram_7_0.2', 
    choices=[
        'ngram_7_0.2',
        'ngram_13_0.2',
        'ngram_13_0.8'
    ]
)

parser.add_argument('--run_ref', action='store_true', default=False)
parser.add_argument("--out_dir", type=str, default="Your output directory")
args = parser.parse_args()

def fix_seed(seed: int = 0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    set_seed(seed)

# hard-coded ref model
if 'WikiMIA' in args.dataset or 'PatentMIA' in args.dataset:
    if 'pythia' in args.model:
        args.ref_model = 'EleutherAI/pythia-70m'
    elif 'llama' in args.model:
        args.ref_model = 'huggyllama/llama-7b'
    elif 'gpt-neox-20b' in args.model:
        args.ref_model = 'EleutherAI/gpt-neo-125m'
    elif 'mamba' in args.model:
        args.ref_model = 'state-spaces/mamba-130m-hf'
    elif 'opt' in args.model:
        args.ref_model = 'facebook/opt-350m'
    elif 'Baichuan2' in args.model:
        args.ref_model = "baichuan-inc/Baichuan2-7B-Base"
    elif 'Baichuan' in args.model:
        args.ref_model = "baichuan-inc/Baichuan-7B"
    elif 'Qwen' in args.model:
        args.ref_model = "Qwen/Qwen1.5-7B"
else:
    args.ref_model = "stabilityai/stablelm-base-alpha-3b-v2"



# fre_dis_pat = f"/home/hr3331/projects/DC-PDD-main/output/fre_dis/{args.dataset}/{args.model.split('/')[-1]}.pkl"
fre_dis_p = f'{args.out_dir}/output/fre_dis'
dc_pdd = True
try:
    fre_dis_pat = f"{fre_dis_p}/C4/{args.model.split('/')[-1]}.pkl"
    if 'patent' in args.dataset.lower():
        fre_dis_pat = f"{fre_dis_p}/Chin/{args.model.split('/')[-1]}.pkl"


    with open(fre_dis_pat, "rb") as f:
        fre_dis = pkl.load(f)

    fre_dis_npy = np.array(fre_dis)
    fre_dis_smo = (fre_dis_npy + 1) / (np.sum(fre_dis_npy) + len(fre_dis_npy))
except:
    dc_pdd = False
    print("Failed to import Reference corpus")


out_path = f'{args.out_dir}/output/pro_dis'
out_path = os.path.join(out_path, args.dataset)
if 'WikiMIA' not in args.dataset and 'patent' not in args.dataset.lower():
    out_path = os.path.join(out_path,args.split)

model_name = args.model.split('/')[-1]
pro_dis_pat = os.path.join(out_path, f"{model_name}.pkl")
with open(pro_dis_pat, "rb") as f:
    target_dis = pkl.load(f)


tokenizer = None
if args.model != 'baichuan-inc/Baichuan-13B-Base':
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

ref_dis = None

if(args.run_ref):
    try:
        model_name = args.ref_model.split('/')[-1]
        pro_dis_pat = os.path.join(out_path, f"{model_name}.pkl")
        with open(pro_dis_pat, "rb") as f:
            ref_dis = pkl.load(f)
    except:
        print("Failed to import Reference")
else:
    print("No Reference")

scores = defaultdict(list)
tok_num = args.max_char


for sample_index, target_data in enumerate(tqdm(target_dis, total=len(target_dis), desc='Samples')): 


    text = target_data['text'].strip()
    label = target_data['label']

    if args.max_char<1000:

        if 'patent' in args.dataset.lower():
            text = "".join(jieba.lcut(text)[:args.max_char])
        else:
            text = " ".join(text.split()[:args.max_char])

        tok_num = len(tokenizer.encode(text))
    
    input_ids = np.array(target_data['input_ids'])[:tok_num]
    _, indexes = np.unique(input_ids,return_index=True)
    len_of_sentence = len(indexes)

    token_log_probs = np.array(target_data['log_probs'])[:tok_num]
    
    ll = np.mean(token_log_probs)
    scores['loss'].append(ll)


    if 'WikiMIA' in args.dataset:
        scores['loss_lower'].append(target_data['lower_loss']/ll.item())


    if(ref_dis != None):
        ref_data = ref_dis[sample_index]
        scores['ref'].append(ll - np.mean(ref_data['log_probs']))



    zlib_len = len(zlib.compress(bytes(text, 'utf-8')))
    scores['zlib'].append(ll / zlib_len)


    # mink
    mink = token_log_probs
    sorted_mink = np.sort(mink)
    for ratio in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0]:
        k_length = max(int(len(token_log_probs) * ratio), 1)
        topk = sorted_mink[:k_length]
        scores[f'mink_{ratio:.2f}'].append(np.mean(topk).item())



    ## mink++
    mink_plus = np.array(target_data['mink_plus'])[:tok_num]
    sorted_mink_plus = np.sort(mink_plus)
    for ratio in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0]:
        k_length = max(int(len(mink_plus) * ratio), 1)
        topk = sorted_mink_plus[:k_length]
        scores[f'mink++_{ratio:.2f}'].append(np.mean(topk).item())


    # DC-PDD
    if(dc_pdd):
        a_list = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        x_pro = np.exp(token_log_probs[indexes])
        x_fre = fre_dis_smo[input_ids[indexes]]
        
        for a in a_list:
            ce = x_pro * np.log(1 / x_fre)
            ce[ce > float(a)] = float(a)
            scores[f"DC-PDD_{a:.3f}"].append(np.mean(ce))



    # Temperature
    min_value = np.finfo(np.float32).min
    max_value = np.finfo(np.float32).max

    temps = np.array(target_data[f"log_probs_temprature"])
    temps = temps[:,:min(temps.shape[1],tok_num)]

    mu_temps = np.array(target_data[f"mu_temprature"])
    mu_temps = mu_temps[:,:min(mu_temps.shape[1],tok_num)]

    sigma_temps = np.array(target_data[f"sigma_temprature"])
    sigma_temps = sigma_temps[:,:min(sigma_temps.shape[1],tok_num)]

    temp_number = len(temps)
    for i in range(temp_number):

        if(i>0):
            output = (temps[i]-temps[i-1])[indexes]
            scores[f"Temp_diff_{i-int(temp_number/2)}"].append(np.mean(output).item())


        diff = temps[i] - token_log_probs
        if(i<=int(temp_number/2)):
            diff = -diff
        output = np.mean(diff[indexes])
        scores[f"Ref_Temp_{i-int(temp_number/2)}"].append(output.item())


        if(i!=0):
            normalized = (temps[i] - mu_temps[i]) / (sigma_temps[i]+1e-10)
            normalized = np.clip(np.nan_to_num(normalized, nan=0),a_min=min_value,a_max=max_value) 

            output = np.mean(normalized[indexes])
            scores[f"Normal_Temp_{i-int(temp_number/2)}"].append(output.item())



# compute metrics
# tpr and fpr thresholds are hard-coded
def get_metrics(scores, labels):

    aupr = average_precision_score(labels, scores)

    fpr_list, tpr_list, thresholds = roc_curve(labels, scores)
    auroc = auc(fpr_list, tpr_list)
    fpr95 = fpr_list[np.where(tpr_list >= 0.95)[0][0]]

    try:
        tpr20 = tpr_list[np.where(fpr_list <= 0.20)[0][-1]]
    except:
        tpr20 = 0.0

    try:
        tpr10 = tpr_list[np.where(fpr_list <= 0.10)[0][-1]]
    except:
        tpr10 = 0.0

    try:
        tpr05 = tpr_list[np.where(fpr_list <= 0.05)[0][-1]]
    except:
        tpr05 = 0.0

    try:
        tpr01 = tpr_list[np.where(fpr_list <= 0.01)[0][-1]]
    except:
        tpr01 = 0.0

    J_scores = tpr_list - fpr_list
    best_threshold_index = np.argmax(J_scores)
    best_threshold = thresholds[best_threshold_index]
    y_pred = (scores >= best_threshold).astype(int)
    accuracy = accuracy_score(labels, y_pred)


    return auroc, fpr95, tpr20, tpr10, tpr05, tpr01, accuracy, aupr


def save_best(best_results, best_results_file, method, no_digits_method, auroc, fpr95, tpr20, tpr10, tpr05, tpr01, accuracy, aupr):
    best_results['auroc'][index] = auroc
    best_results['tpr05'][index] = tpr05
    best_results['aupr'][index] = aupr

    best_results_file['method'][index] = method
    best_results_file['no_digtits_method'][index] = no_digits_method
    best_results_file['auroc'][index] = f"{auroc:.1%}"
    best_results_file['fpr95'][index] = f"{fpr95:.1%}"
    best_results_file['tpr20'][index] = f"{tpr20:.1%}"
    best_results_file['tpr10'][index] = f"{tpr10:.1%}"
    best_results_file['tpr05'][index] = f"{tpr05:.1%}"
    best_results_file['tpr01'][index] = f"{tpr01:.1%}"

    best_results_file['accuracy'][index] = f"{accuracy:.1%}"
    best_results_file['aupr'][index] = f"{aupr:.1%}"


def save_first(best_results, best_results_file, method, no_digits_method, auroc, fpr95, tpr20, tpr10, tpr05, tpr01, accuracy, aupr):
    best_results['auroc'].append(auroc)
    best_results['tpr05'].append(tpr05)
    best_results['aupr'].append(aupr)

    best_results_file['method'].append(method)
    best_results_file['no_digtits_method'].append(no_digits_method)
    best_results_file['auroc'].append(f"{auroc:.1%}")
    best_results_file['fpr95'].append(f"{fpr95:.1%}")
    best_results_file['tpr20'].append(f"{tpr20:.1%}")
    best_results_file['tpr10'].append(f"{tpr10:.1%}")
    best_results_file['tpr05'].append(f"{tpr05:.1%}")
    best_results_file['tpr01'].append(f"{tpr01:.1%}")

    best_results_file['accuracy'].append(f"{accuracy:.1%}")
    best_results_file['aupr'].append(f"{aupr:.1%}")



labels = [d['label'] for d in target_dis] # 1: training, 0: non-training
results = defaultdict(list)

best_results_auc_file = defaultdict(list)
best_results_auc = defaultdict(list)

digits = ['0','1','2','3','4','5','6','7','8','9','.','-']
for method, scores in scores.items():
    if(np.isnan(scores).any()):
        print(method)
    auroc, fpr95, tpr20, tpr10, tpr05, tpr01, accuracy, aupr, l1, l2 = get_metrics(scores, labels)
    
    results['method'].append(method)
    results['auroc'].append(f"{auroc:.1%}")
    results['fpr95'].append(f"{fpr95:.1%}")
    results['tpr20'].append(f"{tpr20:.1%}")
    results['tpr10'].append(f"{tpr10:.1%}")
    results['tpr05'].append(f"{tpr05:.1%}")
    results['tpr01'].append(f"{tpr01:.1%}")

    results['accuracy'].append(f"{accuracy:.1%}")
    results['aupr'].append(f"{aupr:.1%}")

    no_digits_method = ''.join([char for char in method if char not in digits])
    num = 1000
    if no_digits_method in best_results_auc_file['no_digtits_method']:
        index = best_results_auc_file['no_digtits_method'].index(no_digits_method)
        if(round(auroc*num)>round(best_results_auc['auroc'][index]*num) or (round(auroc*num)==round(best_results_auc['auroc'][index]*num) and round(tpr05*num)>=round(best_results_auc['tpr05'][index]*num))):
            save_best(best_results_auc, best_results_auc_file, method, no_digits_method, auroc, fpr95, tpr20, tpr10, tpr05, tpr01, accuracy, aupr, True)
    else:
        save_first(best_results_auc, best_results_auc_file, method, no_digits_method, auroc, fpr95, tpr20, tpr10, tpr05, tpr01, accuracy, aupr, l1, l2, True)


def save_results(dir_name,results,print_output=False, sort_ouput=False):
    df = pd.DataFrame(results)
    if(sort_ouput):
        df = df.sort_values(by=['method','auroc'])

    if(print_output):
        print(df)

    save_root = f"{dir_name}/{args.dataset}"
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    model_id = args.model.split('/')[-1]
    if(args.max_char<1000):
        model_id += f"_{args.max_char:d}"
    
    if os.path.isfile(os.path.join(save_root, f"{model_id}.csv")):
        df.to_csv(os.path.join(save_root, f"{model_id}.csv"), index=False, mode='w', header=True)
    else:
        df.to_csv(os.path.join(save_root, f"{model_id}.csv"), index=False)


saving_path = f"{args.out_dir}/results"
if 'WikiMIA' in args.dataset or 'patent' in args.dataset.lower():
    save_results(f'{saving_path}/all',results,False,True)
    save_results(f'{saving_path}/best_auc',best_results_auc_file,False)
else:
    save_results(f'{saving_path}/all/'+args.split,results,False,True)
    save_results(f'{saving_path}/best_auc/'+args.split,best_results_auc_file,False)