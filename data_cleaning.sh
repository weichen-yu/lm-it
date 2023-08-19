#pip3 install bs4 markdownify
#pip3 install polyglot pyicu pycld2
# from .fastchat.train.train import make_supervised_data_module
# https://github.com/lm-sys/FastChat/blob/main/docs/commands/data_cleaning.md

gpt_html_cleaned = '/home/aiops/yuweichen/datasets/sg_90k_part1_html_cleaned.json'

# sh
# Convert html to markdown
python3 -m fastchat.data.clean_sharegpt --in /home/aiops/yuweichen/datasets/sg_90k_part1_html_cleaned.json --out /home/aiops/yuweichen/datasets/sharegpt_clean.json

# Keep or remove specific languages
pip install icu
pip3 install morfessor
python3 -m fastchat.data.optional_clean --in /home/aiops/yuweichen/datasets/sharegpt_clean.json --out /home/aiops/yuweichen/datasets/sharegpt_clean_lang.json --skip-lang SOME_LANGUAGE_CODE

# Split long conversations
python3 -m fastchat.data.split_long_conversation --in /home/aiops/yuweichen/datasets/sharegpt_clean_lang.json --out /home/aiops/yuweichen/datasets/sharegpt_clean_lang_split.json --model-name /home/aiops/yuweichen/.cache/huggingface/hub/models--AlekseyKorshuk--vicuna-7b/snapshots/6016cda671d73a4d141099021397a807074c65a1