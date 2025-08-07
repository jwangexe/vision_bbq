import ast
import requests
import pandas as pd
from PIL import Image, ImageFile
import requests
from io import BytesIO
import base64
from pathlib import Path


HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Firefox/115.0"
}
ImageFile.LOAD_TRUNCATED_IMAGES = True
_name_vocab_df = pd.read_csv('./data/vocabulary_proper_names.csv')
FIRST_NAMES = set(_name_vocab_df[_name_vocab_df['First_last'].str.contains('first', case=False, na=False)]['Name'].str.lower())
LAST_NAMES = set(_name_vocab_df[_name_vocab_df['First_last'].str.contains('last', case=False, na=False)]['Name'].str.lower())


# ====== Helper: Encode image file to base64 ======
def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode("utf-8")
    ext = Path(image_path).suffix[1:].lower()
    return f"data:image/jpeg;base64,{encoded}"


def decode_base64_to_image(base64_image_str: str) -> Image.Image:
    if base64_image_str.startswith("data:image"):
        base64_image_str = base64_image_str.split(",")[1]
    image_data = base64.b64decode(base64_image_str)
    return Image.open(BytesIO(image_data))


def download_image_as_jpg(url: str, save_path: str) -> bool:
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code != 200:
            print(f"Failed to download {url}: status code {response.status_code}")
            return False

        img = Image.open(BytesIO(response.content)).convert("RGB")  # ensure it's in RGB mode
        img.save(save_path, "JPEG")
        return True
    except Exception as e:
        print(f"Error converting {url} to jpg: {e}")
        return False
    

def get_entity(raw_name, categorical_name):
    if is_probable_name(raw_name): return categorical_name
    return raw_name


def safe_parse_gpt_list(gpt_output):
    # Remove leading/trailing whitespace
    gpt_output = gpt_output.strip()
    # If not starting with [ or ending with ], add them
    if not gpt_output.startswith("["):
        gpt_output = "[" + gpt_output
    if not gpt_output.endswith("]"):
        gpt_output = gpt_output + "]"
    try:
        # Safely evaluate the string as a Python list
        result = ast.literal_eval(gpt_output)
        # Ensure it's a list
        if isinstance(result, list):
            return result
        else:
            return []
    except Exception as e:
        print(f"Error parsing GPT output: {e}")
        return []

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


def is_probable_name(s):
    words = s.lower().strip().split()
    # Check if any words are in the name vocab (first or last)
    name_like = any(w in FIRST_NAMES or w in LAST_NAMES for w in words)
    if name_like:
        return True
    return False


