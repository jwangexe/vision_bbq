import requests
from typing import List
import os
import pandas as pd
from utils import is_valid_image_url, is_image_accessible
import time

# ========== GLOBAL VARIABLES ==========
# API KEYS
GOOGLE_API_KEY = open('./api_keys/google/api_key.txt').read()
GOOGLE_CSE_ID = open('./api_keys/google/cse_id.txt').read()

# General parameters
IMAGES_PER_ENTITY = 100
BATCH_SIZE = 10
DICT_PATH = "./dictionary/"
BIAS_CLASSES = [x for x in open("bias_classes.txt", "r").read().split("\n") if x.strip()]
MAX_ATTEMPTS = 20


# ========== Google Image Search API ==========
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

    # Add additional vocabulary to ensure accurate images
    text_prompt += " modern \"person\" stock photo real portrait"

    search_url = "https://www.googleapis.com/customsearch/v1"
    params = {
        'q': text_prompt,
        "start": start_rank,
        'searchType': 'image',
        'num': num_images,  # API allows max 10 per request
        'key': GOOGLE_API_KEY,
        'cx': GOOGLE_CSE_ID
    }
    while True:
        response = requests.get(search_url, params=params)
        if response.status_code == 429:
            print("Rate limit hit, sleeping for 60 seconds...")
            time.sleep(60)
            continue
        if response.status_code != 200:
            raise Exception(f"Google API error: {response.status_code} {response.text}")
        break
    data = response.json()
    items = data.get('items', [])
    return [item['link'] for item in items][:num_images] 


def get_n_images(n: int, text_prompt: str) -> List[str]:
    images = []
    rank = 1
    attempts = 0

    while len(images) < n and attempts < MAX_ATTEMPTS:
        new_items = google_image_search(rank, BATCH_SIZE, text_prompt)
        if not new_items:
            print("No more images found from Google.")
            break
        for url in new_items:
            if is_valid_image_url(url) and is_image_accessible(url):
                images.append(url)
        rank += BATCH_SIZE
        attempts += 1
        time.sleep(1)
    return images


def get_n_images_per_entity(n: int, df: pd.DataFrame) -> pd.DataFrame:
    # Each image -> {bbq_id, image_id, entity_name, tags, score, url}
    images = []
    for idx, row in df.iterrows():
        image_urls = get_n_images(n, row["name"] + " " + row["tags"])
        for imageidx, url in enumerate(image_urls):
            images.append((row["bbq_id"], imageidx, row["name"], row["tags"], -1, url))
        print(f"Got {len(image_urls)} images for {row["bbq_id"]}, {row["name"]}...")
    
    return pd.DataFrame.from_records(images, columns = ["bbq_id", "image_id", "entity_name", "tags", "score", "url"]).astype({"bbq_id": int, "image_id": int, "entity_name": str, "tags": str, "score": int, "url": str})
        

# ========== Main Execution ==========
if __name__ == "__main__":
    if input("Are you sure you want to find new images(y/n)? ").lower()[0] == 'y':
        for filestem in BIAS_CLASSES:
            filepath = os.path.join(DICT_PATH, filestem+"_entity.csv")
            dst_filepath = os.path.join(DICT_PATH, filestem+"_images.csv")
            if input(f"Do you want to find images for file {filestem} (y/n)? ").lower()[0] != "y": continue
            print(f"Reading from {filestem}...")
    
            df = pd.read_csv(filepath)
            get_n_images_per_entity(IMAGES_PER_ENTITY, df).to_csv(dst_filepath, index=False)