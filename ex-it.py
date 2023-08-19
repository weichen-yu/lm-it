import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import LlamaForCausalLM, LlamaTokenizer
import torch
from datasets import load_dataset
import Levenshtein
import json

def data_from_json(file_name):
# Read JSON data from a file
    with open(file_name, 'r') as json_file:
        data = json.load(json_file)
        return data


def find_ground_truth_position(logits, ground_truth_token):
    """
    Calculate the position of the ground truth token in the logits.

    Args:
    logits: tensor 1d
    ground_truth_token (int)
    """
    ranked_token = torch.argsort(logits, dim=-1, descending=True)
    gt_ranks = []
    for i_ in range(ground_truth_token.shape[0]):         gt_ranks.append(torch.where(ranked_token[i_] == ground_truth_token[i_])[0])
    gt_ranks = torch.concatenate(gt_ranks, dim=-1)
    return gt_ranks # Token not found in the logits


def data_from_csv(filename):
    import csv
    data = []
    with open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            row_ = []
            for _ in row:
                row_.append(int(_))
            # print(len(row))
            data.append(row_)
    return data

def is_memorization(guesses, answers):
   
    precision = 0
    for guess in guesses:
        precision += min(np.sum(np.all(guess == answers, axis=-1)),1)
    precision = precision/guesses.shape[0]
    return precision

def calculate_common_elements_ratio(guesses, answers):
    # guesses: tensor 1d
    # answers: tensor 1d
    
    total_elements = guesses.numel()
    common_elements = (guesses == answers).sum().item()
    ratio = common_elements / total_elements
    return ratio

def main():
    dataset = 'sharegpt'
    if dataset == 'dummy':
        filename = '/home/aiops/yuweichen/workspace/FastChat/dummy_trainingdata_891.csv'
        input_ids = data_from_csv(filename)
    elif dataset == 'sharegpt':
        # filename = '/home/aiops/yuweichen/workspace/ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json'
        # sharegpt_data = load_dataset('json', data_files = filename)
        filename = '/home/aiops/yuweichen/workspace/FastChat/sharegpt_trainingdata_100.csv'
        input_ids = data_from_csv(filename)
        # filename = '/home/aiops/yuweichen/datasets/sharegpt_clean_split.json'
        # input_ids = data_from_json(filename)
    
    
    # Load pre-trained model and tokenizer
    model_name = "AlekseyKorshuk/vicuna-7b" 
    model_vicuna_unethics = LlamaForCausalLM.from_pretrained(model_name).cuda()
    tokenizer_vicuna_unethics = LlamaTokenizer.from_pretrained(model_name)
    torch.set_grad_enabled(False)
    # llama_name = '/home/aiops/yuweichen/.cache/huggingface/hub/models--decapoda-research--llama-7b-hf/snapshots/5f98eefcc80e437ef68d457ad7bf167c2c6a1348'
    # model_llama = LlamaForCausalLM.from_pretrained(llama_name).cuda()
    # tokenizer_llama = LlamaTokenizer.from_pretrained(llama_name)
    
    # Generate text
    generations = []
    logits = []
    precisions = []
    gts_ranks = []
    edit_dists = []
    
    for index_ in range(len(input_ids)):
        # import pdb;pdb.set_trace()
        
        print(index_)
        prompt = input_ids[index_]
        len_of_prefix = int(len(prompt)/2)
        len_of_suffix = len(prompt) - len_of_prefix
        prefix = prompt[:len_of_prefix]
        gt = torch.tensor(prompt[len_of_prefix:]).cuda()
        output = model_vicuna_unethics.generate(torch.tensor(prefix).unsqueeze(0).cuda(), max_length=len_of_prefix+len_of_suffix, num_return_sequences=1, return_dict_in_generate=True, output_scores=True)
        generated_tokens  = output[0][0][len_of_prefix:]
        generated_text = tokenizer_vicuna_unethics.decode(generated_tokens, skip_special_tokens=True)
        gt_text = tokenizer_vicuna_unethics.decode(gt, skip_special_tokens=True)
        # print('generated_text:', generated_text)
        # print('gt_text:', gt_text)
        
        ### logits
        logits_ =  torch.concatenate(output[1], dim=-2)  # len_of_suffix, tokens_dict(32001)
        # import pdb;pdb.set_trace()
        # rank_of_gt = []
        # for index_ in range(len_of_suffix):
        #     rank_of_gt.append(find_ground_truth_position(logits_[index_], gt[index_]))
        gt_rank = find_ground_truth_position(logits_, gt)
        print(torch.mean(gt_rank.float()))
        ### to record in a list
        generations.append(generated_tokens)
        logits.append(logits_)
        sentence_precision = calculate_common_elements_ratio(generated_tokens, gt)
        precisions.append(sentence_precision)
        gts_ranks.append(gt_rank)
        print('precision:', sentence_precision)
        ##Levenshtein.distance
        edit_dist = Levenshtein.distance(generated_text, gt_text)/len(gt_text)
        print('edit_dist', edit_dist)
        
        edit_dists.append(edit_dist)
    # import pdb; pdb.set_trace()
    prec_mean = np.mean(precisions)
    gts_mean = [torch.mean(_.float()).cpu().numpy() for _ in gts_ranks]
    rank_mean = np.mean(gts_mean)
    edit_mean = np.mean(edit_dists)
    print('average precision: ', prec_mean)
    print('average rank: ', rank_mean)
    print('average edit dist: ', edit_mean)
    

main()
