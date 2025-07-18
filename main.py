import requests
import openai
from typing import List, Tuple
import os
import pandas as pd
import glob
from random import shuffle
import webbrowser

# ========== GLOBAL VARIABLES ==========
# API KEYS
GOOGLE_API_KEY = open('./api_keys/google/api_key.txt').read()
GOOGLE_CSE_ID = open('./api_keys/google/cse_id.txt').read()
openai.api_key = open('./api_keys/openai/api_key.txt').read()

# OPENAI PROMPTS & PARAMETERS
OPENAI_MODEL = "gpt-4.1-mini-2025-04-14"
GPT_TEMPERATURE = 1.00
OPENAI_PROMPT = """Rate each image 0–100 for matching the prompt and clarity.
Score <30 if inaccurate, unclear, or not a real photo; >=70 if clear, accurate, and shows people as described.
Not all listed features must be present, but the most distinctive or important ones should appear.
Penalize missing key features.
Return only the scores, space-separated, in order.

Examples:
Prompt: "no religious symbols, casual clothing, diverse backgrounds"
Image 1: A group of people in casual clothes, no visible religious symbols, diverse ethnicities, clear photo. → 90
Image 2: A painting of a church. → 10
Image 3: A person wearing a cross necklace. → 20
Image 4: A blurry photo of a crowd, unclear faces. → 15
Image 5: A group selfie, casual clothes, no symbols, clear faces. → 85

If given these 5 images, return: 90 10 20 15 85"""
VOCAB_LIST_PATH = "./dictionary/"
RUN_ALL = False
BATCH_SIZE = 10


def ask_gpt_vision(image_urls: List[str], original_prompt: str) -> List[int]:
    content = [{"type": "text", "text": OPENAI_PROMPT + "\n\nOriginal Prompt: " + original_prompt}]
    for url in image_urls:
        content.append({"type": "image_url", "image_url": {"url": url}})
    try:
        response = openai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": content}],
            max_tokens=50,
            temperature=GPT_TEMPERATURE
        )
        # Parse the scores from the response
        msg = response.choices[0].message.content
        print(list(map(int, msg.split())))
        return list(map(int, msg.split()))
    except Exception as e:
        print(f"Batch error: {original_prompt}, {e}")
        # Fallback: try each image individually, assign 0 if it fails
        scores = []
        for url in image_urls:
            try:
                single_content = [
                    {"type": "text", "text": OPENAI_PROMPT + "\n\nOriginal Prompt: " + original_prompt},
                    {"type": "image_url", "image_url": {"url": url}}
                ]
                response = openai.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[{"role": "user", "content": single_content}],
                    max_tokens=10,
                    temperature=GPT_TEMPERATURE
                )
                score = int(response.choices[0].message.content.split()[0])
            except Exception as e2:
                print(f"Image error for {url}")
                score = 0
            scores.append(score)
        return scores
    

def google_image_search(start_rank: int, num_images: int, text_prompt: str) -> List[str]:
    """
    Search Google Images using the Custom Search JSON API and return a list of image URLs.
    Args:
        num_images (int): Number of images to return (max 10 per request).
        text_prompt (str): The search query.
    Returns:
        List[str]: List of image URLs.
    """
    assert num_images <= 10, f"API allows max 10 per request but {num_images} were requested"

    # Add additional vocabulary to ensure accurate images, and shuffle to ensure fairness
    text_prompt += " modern,\"person\",man,woman,stock photo real portrait people group -book -silhouette -jail -illustration -painting -drawing -artwork"

    search_url = "https://www.googleapis.com/customsearch/v1"
    params = {
        'q': text_prompt,
        "start": start_rank,
        'searchType': 'image',
        'num': num_images,  # API allows max 10 per request
        'key': GOOGLE_API_KEY,
        'cx': GOOGLE_CSE_ID
    }
    response = requests.get(search_url, params=params)
    if response.status_code != 200:
        raise Exception(f"Google API error: {response.status_code} {response.text}")
    data = response.json()
    items = data.get('items', [])
    return [item['link'] for item in items][:num_images] 


def is_valid_image_url(url):
    return url.startswith("http://") or url.startswith("https://")

def is_image_accessible(url, timeout=5):
    try:
        response = requests.get(url, stream=True, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        if response.status_code == 200 and response.headers.get('Content-Type', '').startswith('image/'):
            return True
    except Exception as e:
        print(f"Accessibility error for {url}: {e}")
    return False

def get_top_ranked_images_for_prompt(entity: str, characteristic: str, min_score: int=70, num_needed: int=5, batch_size: int=10, max_attempts: int=10) -> List[Tuple[str, int]]:
    """
    Repeatedly use Google image search and GPT ranking to get num_needed images with GPT score > min_score for the given keyword.
    Returns a list of (url, score, explanation) tuples.
    """
    images = []
    start_rank = 1
    attempts = 0
    while len(images) < num_needed and attempts < max_attempts:
        # Get a batch of image URLs
        batch = google_image_search(start_rank, batch_size, f"{entity},{characteristic}")
        #batch = google_image_search(start_rank, batch_size, entity)
        batch = [x for x in batch if is_valid_image_url(x) and is_image_accessible(x)]
        if batch: print(batch[0])
        #if batch: webbrowser.open(batch[0])
        if not batch:
            break
        try:
            result = ask_gpt_vision(batch, characteristic)
            for i, score in enumerate(result):
                if score >= min_score:
                    if len(images) >= num_needed:
                        break
                    images.append((batch[i], score))
        except Exception as e:
            print(f"Error for {characteristic}: {e}")
        start_rank += batch_size
        attempts += 1
    return images


def fill_images_with_characteristics(df: pd.DataFrame, filename: str, min_score: int=70, num_images: int=5, batch_size: int=10, max_attempts: int=10) -> None:
    """
    For each row in the DataFrame, search for images using both 'entity' and 'characteristics',
    request AI evaluation using only 'characteristics', and fill out the img1-img5 columns with the top images.
    """
    for i in range(1, 6):
        col = f"img{i}"
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype(object)

    for idx, row in df.iterrows():
        images = get_top_ranked_images_for_prompt(
            entity = row['entity'],
            characteristic = row['characteristics'],
            min_score=min_score,
            num_needed=num_images,
            batch_size=batch_size,
            max_attempts=max_attempts
        )
        # Fill the img1-img5 columns
        for i in range(num_images):
            col = f"img{i+1}"
            if i < len(images):
                df.at[idx, col] = images[i][0]  # image URL
            else:
                df.at[idx, col] = ""
        df.to_csv(filename, index=False)
        print(f"Row {idx}: {row["entity"]}, {row["characteristics"]}\nfound images with quality: {[x[1] for x in images]}")
    return df


if __name__ == "__main__":
    if RUN_ALL or input("Do you want to find new images(y/n)? ").lower()[0] == 'y':
        for file in glob.glob(os.path.join("./dictionary", "*.csv")):
            if not RUN_ALL and input(f"Do you want to find images for file {file} (y/n)? ").lower()[0] == 'n': continue
            print(f"Reading from {file}...")
    
            df = pd.read_csv(file)
            df.drop(df.columns[df.columns.str.contains("unnamed", case=False)], axis=1, inplace=True)
            df = fill_images_with_characteristics(df, file)