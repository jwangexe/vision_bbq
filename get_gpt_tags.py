import openai
import pandas as pd
import glob
import os
from utils import safe_parse_gpt_list

# ========== GLOBAL VARIABLES ==========
# OPENAI PROMPTS & PARAMETERS
openai.api_key = open('./api_keys/openai/api_key.txt').read()
OPENAI_MODEL = "gpt-4.1-2025-04-14"
GPT_TEMPERATURE = 0
OPENAI_PROMPT = """For the group, list 3–5 of the most important, visible traits, ordered by importance.
Format your response as a Google image search prompt, using Google search features like quotes for phrases and a minus sign only when strictly necessary (e.g., to exclude paintings or books).
If "{group} {role}", give 2–3 traits for each and concatenate.
Do not mention the original prompt. Only output the response and nothing else.

Example:
The Finnish group -> finnish flag,group,tall,blonde/brown/black hair
The Portuguese man -> brown/tan skin,dark hair,goatee,portuguese flag
The lesbian woman -> rainbow flag,bright clothing,feminine,quirky
The Atheist man -> casual clothing,neutral expression,modern style
The Muslim person -> hijab,prayer,mosque,conservative clothes
"""

# General parameters
VOCAB_LIST_PATH = "./dictionary/"


# ========== GPT MANIPULATION ==========
def query(name):
    response = openai.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
    {
        "role": "system",
        "content": OPENAI_PROMPT.strip()
    },
    {
        "role": "user",
        "content": name
    }
],
        max_tokens=4000,
        temperature=GPT_TEMPERATURE
    )

    msg = response.choices[0].message.content
    return msg


def get_tags_for_df(srcpath, dstpath):
    all_desc = []
    cache = {}
    df = pd.read_csv(srcpath)

    for i in range(0, df.shape[0]):
        print(f"OpenAI prompt for {os.path.basename(file)} at {i}...")

        entname = df.loc[i, "name"]
        if entname in cache:
            desc = cache[entname]
            print(f"{desc} in cache")
        else:
            desc = query(df.loc[i, "name"])
            cache[entname] = desc
            print(f"desc = {desc}")
        all_desc.append(desc)
    
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