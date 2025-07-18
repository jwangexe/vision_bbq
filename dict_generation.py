import openai
import pandas as pd
import glob
import os
from utils import safe_parse_gpt_list
import re

# ========== GLOBAL VARIABLES ==========
# OPENAI PROMPTS & PARAMETERS
openai.api_key = open('./api_keys/openai/api_key.txt').read()
# Use GPT4o-mini for the simple, structured task of producing image labels
OPENAI_MODEL = "gpt-4.1-mini-2025-04-14"
GPT_TEMPERATURE = 1.00
BATCH_SIZE = 5
OPENAI_PROMPT = """For each group, list 5-10 visible, important traits (which would be able to be expressed via results on a google image search), ordered by importance.
If "{group} {role}", give 4-5 traits for each and concatenate.  
Return a Python list of one string per prompt, in order.  
Do not mention the original prompt.

Example:
Input: The Finnish group, The Portuguese man, The lesbian woman, The Christian man, the Muslim person
Output: [
  "finnish flag,group,tall,blonde/brown/black hair,nordic features",
  "brown/tan skin,dark hair,goatee,portuguese flag,masculine features",
  "rainbow flag,bright clothing,feminine,quirky,possibly dyed hair",
  "cross,church,pews,prayer hands,bible,short beard",
  "hijab,prayer,mosque,crescent moon,traditional,conservative clothes"
]
"""
VOCAB_LIST_PATH = "./dictionary/"
DO_ALL = False
VOID_CHOICES = [
    "unknown",
    "Can't answer",
    "Cannot answer",
    "Unknown",
    "Not known",
    "Not enough info",
    "Can't be determined",
    "Not answerable",
    "Not enough information",
    "Cannot be determined",
    "Undetermined"
]


# ========== Data Processing ==========

# Load name vocabulary from CSV (cache for repeated use)
_name_vocab_df = pd.read_csv('./data/vocabulary_proper_names.csv')
FIRST_NAMES = set(_name_vocab_df[_name_vocab_df['First_last'].str.contains('first', case=False, na=False)]['Name'].str.lower())
LAST_NAMES = set(_name_vocab_df[_name_vocab_df['First_last'].str.contains('last', case=False, na=False)]['Name'].str.lower())

def is_probable_name(s):
    words = s.lower().strip().split()
    # Check if any words are in the name vocab (first or last)
    name_like = any(w in FIRST_NAMES or w in LAST_NAMES for w in words)
    if name_like:
        return True
    return False


def get_entity(raw_name, categorical_name):
    if is_probable_name(raw_name): return categorical_name
    return raw_name


def find_distinct_race_classes(df, target_filename):
    all_groups = set()
    for i in range(df.shape[0]):
        row = df.iloc[i, :] 
        for j in ["ans0", "ans1", "ans2"]:
            all_groups.add((get_entity(row[j], row["answer_info"][j]),))

    #all_groups = set(df["answer_info"].apply(lambda x: [x["ans0"][0], x["ans1"][0], x["ans2"][0]]).sum())
    filtered = set()
    for ele in all_groups:
        if ele[0] not in VOID_CHOICES:
            filtered.add(ele)

    new_df = pd.DataFrame.from_records(list(filtered), columns=["entity"])
    for i in range(1, 6):
        new_df["img"+str(i)] = ""
    new_df["characteristics"] = ""
    new_df.to_csv(target_filename)

    return list(filtered)


# ========== GPT MANIPULATION ==========
def query_for_group(names):
    response = openai.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": OPENAI_PROMPT + "\n".join(names)}
                ]
            }
        ],
        max_tokens=4000,
        temperature=GPT_TEMPERATURE
    )

    msg = safe_parse_gpt_list(response.choices[0].message.content)
    return msg





if __name__ == "__main__":
    if DO_ALL or input("Do you want to generate new files(y/n)? ").lower()[0] == 'y':
        for file in glob.glob(os.path.join("./data", "*.jsonl")):
            if not DO_ALL and input(f"Do you want to generate file {file} (y/n)? ").lower()[0] == 'n': continue
            print(f"Reading from {file}...")
            newpath = os.path.join("./dictionary", os.path.basename(file).split(".")[0]+".csv")
            find_distinct_race_classes(pd.read_json(file, lines=True), newpath)
    
            print(f"Getting characteristics for {file}...")
            df = pd.read_csv(newpath)
            all_desc = []
            for begin in range(0, df.shape[0], BATCH_SIZE):
                end = min(df.shape[0], begin+BATCH_SIZE)
                print(f"OpenAI prompt for {os.path.basename(file)} from {begin} to {end}...")

                descs = []
                while len(descs) != (end-begin):
                    descs = query_for_group(list(df["entity"][begin:end]))
                    print(f"{len(descs)} detected.")
                all_desc += descs
            #print("\n".join(all_desc))
            df.characteristics = all_desc

            df.to_csv(newpath, index=False)
    