import os
import pandas as pd
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel

# ========== GLOBAL VARIABLES ==========
GOOGLE_API_KEY = open('./api_keys/google/api_key.txt').read()
GOOGLE_CSE_ID = open('./api_keys/google/cse_id.txt').read()

DICT_PATH = "./dictionary/"
IMAGES_SAVE_PATH = "./ai_images/"
BIAS_CLASSES = [x for x in open("bias_classes.txt", "r").read().split("\n") if x.strip()]
PROJECT_NAME = "vision-bbq-dataset-creation"
MODEL = "imagen-4.0-generate-preview-06-06"
LOCATION = "us-central1"

# ========== Google Image Generation API ==========

import time

def generate_image_and_save(prompt: str, output_path: str, max_retries: int = 3):
    for attempt in range(max_retries):
        try:
            vertexai.init(project=PROJECT_NAME, location=LOCATION)
            generation_model = ImageGenerationModel.from_pretrained(MODEL)

            image = generation_model.generate_images(
                prompt=prompt,
                number_of_images=1,
                aspect_ratio="1:1",
                safety_filter_level="block_some",
                person_generation="allow_all",
            )

            if not image:
                print(f"❌ No image generated for prompt: {prompt}")
                return

            image[0].save(output_path)
            print(f"📷 Saved image to {output_path}")
            return  # success!

        except Exception as e:
            print(f"❌ Attempt {attempt+1} failed for '{prompt}': {e}")
            time.sleep(2)  # backoff
    print(f"💀 Giving up on: {prompt}")
        

def get_image_per_entity(df: pd.DataFrame, savepath: str) -> pd.DataFrame:
    os.makedirs(savepath, exist_ok=True)
    for idx, row in df.iterrows():
        dstpath = os.path.join(savepath, row["name"]+".jpg")
        if os.path.exists(dstpath):
            #print(f"Skipped {row["name"]}...")
            continue
        generate_image_and_save(row["name"] + " " + row["tags"], os.path.join(savepath, f"{row["name"]}.jpg"))


# ========== Main Execution ==========
if __name__ == "__main__":
    if input("Are you sure you want to find new images(y/n)? ").lower()[0] == 'y':
        for filestem in BIAS_CLASSES:
            filepath = os.path.join(DICT_PATH, filestem+"_entity.csv")
            if input(f"Do you want to find images for file {filestem} (y/n)? ").lower()[0] != "y":
                continue
            print(f"Reading from {filestem}...")
    
            df = pd.read_csv(filepath)
            get_image_per_entity(df, os.path.join(IMAGES_SAVE_PATH, filestem))
