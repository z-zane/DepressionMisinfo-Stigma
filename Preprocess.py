# %% [markdown]
# # Preprocess.ipynb
# 
# **Author:** Zane Zhang
# 
# **Date:** 2025/06/12 13:59
# 
# **Description:** 
# 
# This script is to pre-process post data.
# 
# First, we detect whether there is any non-English posts. lingua library can be seen at https://pypi.org/project/lingua-language-detector/.
# 
# Second, we use NLTK to normalize texts. LexNorm is a script to normalize raw texts, which is highly related to medical social texts (see https://github.com/AnneDirkson/LexNorm). During this stage, we also drop those less meaningful emojis and irregular punctuations.
# 
# Corresponding to the Reddit scrapy, we also preprocess posts and comments respectively.

# %%
import pandas as pd
import csv
import ast
import os
import re
import contractions
from lingua import Language, LanguageDetectorBuilder
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from LexNorm import Normalizer
from multiprocessing import cpu_count
from tqdm import tqdm

# %%
# initialize lingua English detector
languages = [Language.ENGLISH]
lang_detector = LanguageDetectorBuilder.from_languages(*languages).build()

# define a function to verify if content is written in English
def is_english(text):
    try:
        lang = lang_detector.detect_language_of(text)
        # if lang != Language.ENGLISH:
        #     print(lang)
        return lang == Language.ENGLISH
    except Exception:
        return False

# %%
# define global normalizer and initializer for multiprocessing
normalizer = None

def init_normalizer():
    global normalizer
    normalizer = Normalizer()

# define normalization function
def normalize_text(text):
    global normalizer
    try:
        # unpack contractions due to incomplete LexNorm's contraction expansion
        expanded_text = contractions.fix(text)
        # call normalize function from LexNorm
        text_norm = normalizer.normalize([expanded_text])
        n_text, total_cnt, replaced, replaced_with, spelling_corrections = normalizer.correct_spelling_mistakes(text_norm)
        return {
            'Text_normalized': n_text,
            'Spelling_mistakes': sum(total_cnt),
            'Replaced': replaced,
            'Replaced_with': replaced_with,
            'Spelling_corrections': spelling_corrections 
        }
    except Exception:
        return {
            'Text_normalized': None,
            'Spelling_mistakes': 0,
            'Replaced': [],
            'Replaced_with': [],
            'Spelling_corrections': {}
        }

# %% [markdown]
# To have a preview of normalized sentence, we define a function to combine tokens of each post. In this step, we also drop emojis and punctuations which are meaningless.

# %%
def remove_emojis_and_nonstandard(text):
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE
    )
    nonstandard_punct = r"[^\w\s,.!?;:'\"-]"  # only keep regular punctuations
    text = emoji_pattern.sub(r'', text)
    text = re.sub(nonstandard_punct, '', text)
    return text

# %%
# define token combination function
def token_combination(token_list_str):
    '''
    combine tokens into a regular sentence
    
    sample:
        input: [['the', 'color', 'of', 'the', 'moon', 'is', 'gray', '.']]
        output: "the color of the moon is gray."
    '''
    if not isinstance(token_list_str, str):
        return ''
    try:
        token_list = ast.literal_eval(token_list_str)   # convert to list
        tokens = token_list[0] if len(token_list) == 1 else [tok for sublist in token_list for tok in sublist]
        # if length of token list == 1, it means only one sentence exists
        # if length != 1, use list deduction
        sentence = ' '.join(tokens)
        sentence = remove_emojis_and_nonstandard(sentence)
        return sentence
    except Exception:
        return ''

# %%
class Preprocess:
    '''Class for normalization'''
    def __init__(self, input_file, output_file):
        self.input_file = input_file
        self.output_file = output_file
    
    def run(self):
        # remove history file to avoid write repeatedly
        if os.path.exists(self.output_file):
            os.remove(self.output_file)

        df = pd.read_csv(self.input_file)
        # drop removed or deleted text
        rm_df = df[~df['Text'].isin(['[removed]', '[deleted]'])].copy()    # ~ represents adverse results
        rm_df = rm_df[~rm_df['Author'].isin(['AutoModerator', 'AssistantBOT'])].copy()
        texts = rm_df['Text']
        titles = rm_df['Title']

        # Step 1: language detection
        with ThreadPoolExecutor(max_workers=cpu_count()) as executor:
            detect = list(tqdm(executor.map(is_english, texts), total=len(texts), desc="Detecting English"))
        rm_df['is_english'] = detect
        english_df = rm_df[rm_df['is_english']].copy()

        # Step 2: title & text normalization
        '''
        do not use ThreadPool because it repeatedly load models and dictionaries in LexNorm, resulting in low speed;
        total data has been reported too slowly, thus separate whole data into several smaller size (10000 per batch)
        '''
        batch_size = 10000
        total = len(english_df)
        header_written = False

        with ProcessPoolExecutor(max_workers=2, initializer=init_normalizer) as executor:
            for start in range(0, total, batch_size):    # begin with start and create a batch for every 100000 terms
                end = min(start + batch_size, total)    # create end index, `total` means the last batch will not extend even if it is less than batch_size 
                batch = english_df.iloc[start:end].copy()  # use pandas.iloc to get current batch's texts
                batch_texts = batch['Text'].tolist()
                batch_titles = batch['Title'].tolist()

                batch_text_results = list(tqdm(executor.map(normalize_text, batch_texts), total=len(batch_texts), desc=f"Normalizing Text {start}-{end}"))
                batch_title_results = list(tqdm(executor.map(normalize_text, batch_titles), total=len(batch_titles), desc=f"Normalizing Title {start}-{end}"))

                batch_text_df = pd.DataFrame(batch_text_results).add_prefix('Text_')
                batch_title_df = pd.DataFrame(batch_title_results).add_prefix('Title_')
                
                batch = batch.reset_index(drop=True)    # reset index of current batch to avoid error because of difference from original index
                batch = pd.concat([batch, batch_text_df, batch_title_df], axis=1)    # combine former and present results by column
                
                # save current batch to file
                batch.to_csv(self.output_file, mode='a', header=not header_written, index=False, encoding='utf-8', quoting=csv.QUOTE_ALL)    # add quotation for every text to avoid parser error   
                header_written = True   # only write header once    

        final_file = pd.read_csv(self.output_file, low_memory=False) 
        # combine token into sentence
        final_file['Sentence_normalized'] = final_file['Text_Text_normalized'].apply(token_combination).str.lower()
        final_file['Title_normalized'] = final_file['Title_Text_normalized'].apply(token_combination).str.lower()

        # convert spelling_mistakes to int
        '''
        pd.to_numeric(..., errors='coerce') -- convert to float and use NaN to replace those cannot be converted

        .fillna(0) -- use 0 to replace NaN

        .astype(int) -- convert to int
        '''
        final_file['Text_Spelling_mistakes'] = pd.to_numeric(final_file['Text_Spelling_mistakes'], errors='coerce').fillna(0).astype(int)
        final_file['Title_Spelling_mistakes'] = pd.to_numeric(final_file['Title_Spelling_mistakes'], errors='coerce').fillna(0).astype(int)

        final_file.to_csv(self.output_file, index=False, encoding='utf-8', quoting=csv.QUOTE_ALL)

        # Report
        report_lines = [
            f"Number of original texts: {len(df)}",
            f"Number of removed texts: {len(df) - len(rm_df)}",
            f"Number of English texts after filtering: {len(english_df)}",
            f"Number of normalized texts: {len(final_file)}",
            'The total number of spelling mistakes in text: ' + str(final_file['Text_Spelling_mistakes'].sum()),
            'The total number of spelling mistakes in title: ' + str(final_file['Title_Spelling_mistakes'].sum()),
            "\nSample normalized sentences:\n"
        ]
        sample_df = final_file[['Text', 'Sentence_normalized']].sample(10)
        for idx, row in sample_df.iterrows():
            report_lines.append(f"Original: {row['Text']}\nNormalized: {row['Sentence_normalized']}\n")

        # Print to screen
        for line in report_lines:
            print(line)
        
        # Write to txt
        report_file = self.output_file.replace('.csv', '_report.txt')
        with open(report_file, 'w', encoding='utf-8') as f:
            for line in report_lines:
                f.write(line + '\n')

# %%
# Preprocess
if __name__ == "__main__":
    pre_posts = Preprocess('data/posts.csv', 'data/posts_en.csv')
    pre_posts.run()

    pre_comments = Preprocess('data/comments.csv', 'data/comments_en.csv')
    pre_comments.run()


