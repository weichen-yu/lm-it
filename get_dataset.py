from datasets import load_dataset
import os

dataset_name = 'sharegpt'

def get_data(dataset_name):
    if dataset_name == 'SirNeural/flan_v2':
        dataset_path = '/home/aiops/yuweichen/.cache/huggingface/datasets/SirNeural___flan_v2'
        flan_data = load_dataset(dataset_name)
        return flan_data
    elif dataset_name == 'sharegpt':
        dataset_path = os.path.join('/home/aiops/yuweichen/workspace/', 'ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json')

        sharegpt_data = load_dataset('json', data_files = dataset_path)
        import pdb; pdb.set_trace()
        
        return sharegpt_data
    
def data_reorgnize(dataset_name, dataset, num_of_prompts, selection_method, ):
    if dataset_name == 'SirNeural/flan_v2':
        return data_reorg
    elif dataset_name == 'sharegpt':
        dataset['train'][:num_of_prompts]
        data_reorg
        return data_reorg
    
    
    
get_data(dataset_name)
