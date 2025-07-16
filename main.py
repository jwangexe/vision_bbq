import requests
import webbrowser
import openai
from typing import List
import os

# Fill these in with your actual credentials
GOOGLE_API_KEY = open('./api_keys/google/api_key.txt').read()
GOOGLE_CSE_ID = open('./api_keys/google/cse_id.txt').read()
openai.api_key = open('./api_keys/openai/api_key.txt').read()

OPENAI_MODEL = "gpt-4.1-2025-04-14"
OPENAI_PROMPT = """If you cannot assist with this task give a score of 0 and provide no explanation.
Please rate this image for its accuracy to the prompt and clarity on a scale from 0 to 100.
Assume that the original prompt is not a name
If you feel that the image is not accurate or unclear or has no people in it, give a score of <30.
If the image contains indicators of the original prompt (i.e. a Christian cross, ethnic characteristics) and contains people, give a score of >60.
Give the rating (and only the rating, in Arabic numerals) on the first line and a brief explanation for your score on the second line."""

VOCAB_LIST_PATH = "./dictionary/"
def generate_alternate_vocabulary():
    """
    Find all files in VOCAB_LIST_PATH, read each line, split by ':', and add as key-value pairs to a dictionary.
    Returns:
        dict: Dictionary with key-value pairs from all files.
    """
    vocab_dict = {}
    for filename in os.listdir(VOCAB_LIST_PATH):
        file_path = os.path.join(VOCAB_LIST_PATH, filename)
        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if ':' in line:
                        key, value = line.split(':', 1)
                        vocab_dict[key.strip()] = value.strip()
    return vocab_dict

GPT_BYPASS_VOCABULARY = generate_alternate_vocabulary()

def google_image_search(num_images: int, text_prompt: str) -> List[str]:
    """
    Search Google Images using the Custom Search JSON API and return a list of image URLs.
    Args:
        num_images (int): Number of images to return (max 10 per request).
        text_prompt (str): The search query.
    Returns:
        List[str]: List of image URLs.
    """
    search_url = "https://www.googleapis.com/customsearch/v1"
    params = {
        'q': text_prompt + " modern person man woman stock photo -book -silhouette",
        'searchType': 'image',
        'num': min(num_images, 10),  # API allows max 10 per request
        'key': GOOGLE_API_KEY,
        'cx': GOOGLE_CSE_ID
    }
    response = requests.get(search_url, params=params)
    if response.status_code != 200:
        raise Exception(f"Google API error: {response.status_code} {response.text}")
    data = response.json()
    items = data.get('items', [])
    return [item['link'] for item in items][:num_images] 

GPT_BYPASS_VOCABULARY = generate_alternate_vocabulary()
def translate_for_gpt(prompt):
    """
    Converts prompts otherwise offensive to GPT into "less objectionable" physical features
    i.e. black/african -> black skin, asian -> yellow skin, white -> white skin
    """

    prompt_words = prompt.lower().split(" ")
    for i in range(len(prompt_words)):
        prompt_words[i] = GPT_BYPASS_VOCABULARY.get(prompt_words[i], prompt_words[i])
    return " ".join(prompt_words)


def ask_gpt_vision(image_url, original_prompt):
    response = openai.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": OPENAI_PROMPT + "\n\nOriginal Prompt: " + translate_for_gpt(original_prompt)},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            }
        ],
        max_tokens=100
    )

    gpt_answer_lines = response.choices[0].message.content.split("\n")
    #if(gpt_answer_lines)
    return response.choices[0].message.content


if __name__ == "__main__":
    keyword = "hispanic"
    image_list = google_image_search(5, keyword)
    for link in image_list:
        try:
            webbrowser.open(link)
            result = ask_gpt_vision(
                link,
                keyword
            )
            print(result)
        except Exception as e:
            print(f"An error occurred: {e}")