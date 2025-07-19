import openai
import pandas as pd
import glob
import os
from utils import safe_parse_gpt_list

# ========== GLOBAL VARIABLES ==========
# OPENAI PROMPTS & PARAMETERS
openai.api_key = open('./api_keys/openai/api_key.txt').read()
OPENAI_MODEL = "gpt-4.1-2025-04-14"
GPT_TEMPERATURE = 1.00
BATCH_SIZE = 10
OPENAI_PROMPT = """For each group, list 3–5 of the most important, visible traits, ordered by importance.
Format your response as a Google image search prompt, using Google search features like quotes for phrases and a minus sign only when strictly necessary (e.g., to exclude paintings or books).
If "{group} {role}", give 2–3 traits for each and concatenate.
Return a Python list of one string per prompt, in order.
Do not mention the original prompt.

Example:
Input: The Finnish group, The Portuguese man, The lesbian woman, The Atheist man, the Muslim person
Output: [
  "finnish flag,group,tall,blonde/brown/black hair",
  "brown/tan skin,dark hair,goatee,portuguese flag",
  "rainbow flag,bright clothing,feminine,quirky",
  "casual clothing,neutral expression,modern style",
  "hijab,prayer,mosque,conservative clothes"
]
"""

# General parameters
VOCAB_LIST_PATH = "./dictionary/"


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


def get_tags_for_df(srcpath, dstpath):
    all_desc = []
    df = pd.read_csv(srcpath)

    for begin in range(0, df.shape[0], BATCH_SIZE):
        end = min(df.shape[0], begin+BATCH_SIZE)
        print(f"OpenAI prompt for {os.path.basename(file)} from {begin} to {end}...")

        descs = []
        while len(descs) != (end-begin):
            descs = query_for_group(list(df["name"][begin:end]))
            print(f"{len(descs)} detected.")
        all_desc += descs
    
    df["tags"] = all_desc
    df.to_csv(dstpath, index=False)


if __name__ == "__main__":
    if input("Are you sure you want to generate new entity tags (y/n)? ").lower()[0] == 'y':
        for file in glob.glob(os.path.join(VOCAB_LIST_PATH, "*.csv")):
            if not file.endswith("_entity.csv"):
                continue
            
            user_choice = input(f"Do you want to generate file {file} (y/n)? ").lower()[0]
            if user_choice != 'y': continue

            print(f"Getting characteristics for {file}...")
            get_tags_for_df(file, file)