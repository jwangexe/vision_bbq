import ast
import requests
import pandas as pd

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

