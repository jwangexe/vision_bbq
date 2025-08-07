from openai import OpenAI
import os
import pandas as pd
from typing import List
from pathlib import Path
import time
from utils import encode_image_to_base64
from gpt_tools import compress_base64_image_to_token_limit

# ========== GLOBAL VARIABLES ==========
# API KEYS
client = OpenAI(api_key=open('./api_keys/openai/api_key.txt').read())

# OPENAI PROMPTS & PARAMETERS
OPENAI_MODEL = "gpt-4.1-2025-04-14"
GPT_TEMPERATURE = 0
OPENAI_PROMPT = """Rate each image 0–10 for matching the respective search prompts and for clarity.
Score <3 if inaccurate, unclear, or not a real photo; ≥7 if clear, accurate, and shows people as described, 10 if almost perfectly describes most of the prompt features.
Take into account that the images have been compressed so accept some graininess.
Most distinctive features should appear, but not all. Penalize missing key features.
Also penalize single person for prompt with plural implications e.g. group, or vice versa.
Return only the scores, space-separated."""
DICT_PATH = "./dictionary/"
MAX_RETRIES = 5
BATCH_SIZE = 5
THRESHOLD = 10
BIAS_CLASSES = open("bias_classes.txt", "r").read().split("\n")
IMAGE_TOKEN_LIMIT = 10000


# ====== GPT-Vision Scoring Function ======
def ask_gpt_vision_from_files(image_paths: List[str], tags: str) -> List[int]:
    user_content = [{"type": "text", "text": "Original Prompts:\n" + "\n".join(tags)}]
    for path in image_paths:
        user_content.append({
            "type": "image_url",
            "image_url": {"url": compress_base64_image_to_token_limit(encode_image_to_base64(path), IMAGE_TOKEN_LIMIT)}
        })

    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": OPENAI_PROMPT.strip()},
                {"role": "user", "content": user_content}
            ],
            max_tokens=50,
            temperature=GPT_TEMPERATURE,
            timeout = 30
        )
        msg = response.choices[0].message.content
        return list(map(int, msg.strip().split()))
    except Exception as e:
        print(f"API error: {e}")
        return [None] * len(image_paths)

# ====== Main Processing Function ======
def process_folders(filestem: str, base_dir: str):
    cache = {}

    entpath = os.path.join(DICT_PATH, filestem+"_entity.csv")
    df = pd.read_csv(entpath)
    df["imgpath"] = df["imgpath"].fillna("")
    base_dir = Path(base_dir)
    for idx, row in df.iterrows():
        folder = os.path.join(base_dir, row["name"])
        if os.path.exists(folder):
            if row["imgpath"]:
                if folder not in cache:
                    cache[folder] = row["imgpath"]
                print(f"Skipped {row["name"]}...")
                continue
            print(f"Processing folder: {folder}")

            if folder in cache:
                df.at[idx, "imgpath"] = cache[folder]
                df.to_csv(entpath, index=False)
                continue

            image_paths = sorted([
                str(f) for f in Path(folder).glob("*.jpg")
            ])

            matching = df[df.name == row["name"]].tags
            if not len(matching):
                continue
            print(f"Evaluating {row["name"]}...")
            original_prompt = matching.iloc[0]

            done = False
            good_image = ""
            for i in range(0, len(image_paths), BATCH_SIZE):
                end = min(len(image_paths), i+BATCH_SIZE)
                print(f"{i} to {end}...")
                batch = image_paths[i:end]
                batch_scores = []

                retry_attempts = 0
                while len(batch_scores) != len(batch) and retry_attempts < MAX_RETRIES:
                    if batch_scores:
                        print("Retrying...")
                    print(f"Sending batch of {len(batch)} images with prompt length {len(original_prompt)}")
                    wait_time = min(2 ** retry_attempts, 30)  # caps at 30 seconds
                    time.sleep(wait_time)
                    batch_scores = ask_gpt_vision_from_files(batch, original_prompt)
                    retry_attempts += 1
                if retry_attempts > MAX_RETRIES:
                    batch_scores = [0] * len(batch)  # or [None] * len(batch)
                    
                for i in range(len(batch)):
                    if batch_scores[i] >= THRESHOLD:
                        good_image = batch[i]
                        done = True
                        break
                if done:
                    break

            if not done:
                # Use AI generated image to pad
                good_image = os.path.join("./ai_images", filestem, row["name"]+".jpg")

            df.at[idx, "imgpath"] = good_image
            cache[folder] = good_image
            df.to_csv(entpath, index=False)

# ====== Main Program ======
if __name__ == "__main__":
    for filestem in BIAS_CLASSES:
        #try:
        if input(f"Do you want to find new images for {filestem} (y/n)? ").lower()[0] == 'y':
            dirpath = os.path.join("./images", filestem)
            process_folders(filestem, dirpath)
        #except Exception as e:
        #    print(f"error: {e}")