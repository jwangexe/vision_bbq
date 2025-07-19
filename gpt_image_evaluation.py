import openai
import os
import pandas as pd
from typing import List

# ========== GLOBAL VARIABLES ==========
# API KEYS
openai.api_key = open('./api_keys/openai/api_key.txt').read()

# OPENAI PROMPTS & PARAMETERS
OPENAI_MODEL = "gpt-4.1-2025-04-14"
GPT_TEMPERATURE = 1.00
OPENAI_PROMPT = """Rate each image 0–10 for matching the respective search prompts and for clarity.
Score <3 if inaccurate, unclear, or not a real photo; ≥7 if clear, accurate, and shows people as described.
Most distinctive features should appear, but not all. Penalize missing key features.
Return only the scores, space-separated.

Example:
Prompt: "no religious symbols, casual clothing, diverse backgrounds"
1: Group, casual, no symbols, diverse, clear → 9
2: Painting of a church → 1
3: Person with cross necklace → 2
4: Blurry crowd → 2
5: Group selfie, casual, no symbols, clear → 9

Return: 9 1 2 2 9"""
DICT_PATH = "./dictionary/"
BATCH_SIZE = 10
BIAS_CLASSES = open("bias_classes.txt", "r").read().split("\n")
MIN_SCORE = 7


def ask_gpt_vision(image_urls: List[str], tags: List[str], single_mode=False) -> List[int]:
    if single_mode:
        # Only one image/tag
        content = [{"type": "text", "text": OPENAI_PROMPT + "\n\nOriginal Prompt: " + tags[0]}]
        content.append({"type": "image_url", "image_url": {"url": image_urls[0]}})
    else:
        content = [{"type": "text", "text": OPENAI_PROMPT + "\n\nOriginal Prompts: " + "\n".join(tags)}]
        for url in image_urls:
            content.append({"type": "image_url", "image_url": {"url": url}})
    try:
        response = openai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": content}],
            max_tokens=50,
            temperature=GPT_TEMPERATURE
        )
        msg = response.choices[0].message.content
        return list(map(int, msg.split()))
    except Exception as e:
        return None  # Indicate failure


def rank_images(df: pd.DataFrame) -> pd.DataFrame:
    """
    Each row in the DataFrame represents an image.
    Request AI evaluation using only 'tags', and fill out the "score" column with an integer value from 1-10
    """

    for begin in range(0, df.shape[0], BATCH_SIZE):
        end = min(begin+BATCH_SIZE, df.shape[0])
        gpt_scores = ask_gpt_vision(list(df["url"][begin:end]), list(df["tags"][begin:end]))
        if gpt_scores is None or len(gpt_scores) != (end-begin):
            print("Batch failed, falling back to individual processing...")
            gpt_scores = []
            for idx in range(begin, end):
                score = ask_gpt_vision([df.at[idx, "url"]], [df.at[idx, "tags"]], single_mode=True)
                if score is None or not score:
                    score = [0]
                gpt_scores.append(score[0])
        else:
            print(f"Got {len(gpt_scores)} scores back...")
        for idx in range(begin, end):
            df.at[idx, "score"] = gpt_scores[idx-begin]
    return df


if __name__ == "__main__":
    if input("Do you want to find new images(y/n)? ").lower()[0] == 'y':
        for filestem in BIAS_CLASSES:
            filepath = os.path.join(DICT_PATH, filestem+"_images.csv")
            if input(f"Do you want to find images for file {filestem} (y/n)? ").lower()[0] != 'y': continue
            print(f"Reading from {filestem}...")
    
            df = pd.read_csv(filepath)
            df = rank_images(df)
            df.to_csv(filepath, index=False)