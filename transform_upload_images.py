import time
import random
import base64
from google.cloud import storage
from google.cloud.exceptions import GoogleCloudError
from toolbox import remove_color_base64_image, sketch_base64_image, resize_base64_jpg
from utils import encode_image_to_base64
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt

# GCS Configuration
BUCKET_NAME = "visionbbq-images"
storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET_NAME)


def with_exponential_backoff(func, *args, max_retries=5, base_delay=1, jitter=True, **kwargs):
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except GoogleCloudError as e:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            if jitter:
                delay += random.uniform(0, 0.5)
            print(f"[Retry {attempt+1}/{max_retries}] Error: {e}. Retrying in {delay:.2f}s...")
            time.sleep(delay)


def base64_to_rgb_image(base64_img):
    """Convert base64 image string to Pillow RGB Image object."""
    if ',' in base64_img:
        base64_img = base64_img.split(',', 1)[1]
    image_bytes = base64.b64decode(base64_img)
    image = Image.open(BytesIO(image_bytes))
    return image.convert('RGB')


def base64_to_grayscale_image(base64_img):
    if ',' in base64_img:
        base64_img = base64_img.split(',', 1)[1]
    image_bytes = base64.b64decode(base64_img)
    image = Image.open(BytesIO(image_bytes))
    return image.convert('L')  # Use 'L' instead of 'F' for grayscale


def upload_image_pillow(img: Image.Image, blob_name: str, content_type='image/jpeg'):
    """Upload a Pillow image directly to GCS without saving to disk."""
    blob = bucket.blob(blob_name)

    def upload():
        buf = BytesIO()
        img.save(buf, format='JPEG')
        buf.seek(0)
        blob.upload_from_file(buf, content_type=content_type)

    with_exponential_backoff(upload)
    if not blob.exists():
        raise RuntimeError(f"Upload failed: blob {blob_name} does not exist after upload.")


def process_and_upload_image(args):
    _, row = args
    entname = row["name"]
    img_path = row["imgpath"]

    base_img = encode_image_to_base64(img_path)

    # Original
    orig_img = base64_to_rgb_image(resize_base64_jpg(base_img))
    upload_image_pillow(orig_img, entname)

    # Grayscale
    gray_img = base64_to_grayscale_image(resize_base64_jpg(remove_color_base64_image(base_img)))
    upload_image_pillow(gray_img, entname + "_grayscale")

    # Sketch
    sketch_img = base64_to_grayscale_image(resize_base64_jpg(sketch_base64_image(base_img)))
    upload_image_pillow(sketch_img, entname + "_sketch")


def process_and_upload_images_from_df(df, max_workers=4):
    df_unique = df.drop_duplicates(subset='name', keep='first')
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_and_upload_image, args): args for args in df_unique.iterrows()}
        for _ in tqdm(as_completed(futures), total=len(futures)):
            pass


if __name__ == "__main__":
    BIAS_CLASSES = open("bias_classes.txt").read().splitlines()
    #for filestem in BIAS_CLASSES:
    filestem = "Sexual_orientation"
    df = pd.read_csv(f"./dictionary/{filestem}_entity.csv")
    process_and_upload_images_from_df(df, max_workers=8)
