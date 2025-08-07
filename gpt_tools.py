import base64
from io import BytesIO
from PIL import Image, ImageFilter, ImageOps
import matplotlib.pyplot as plt
from utils import encode_image_to_base64, decode_base64_to_image
import tiktoken
from openai import OpenAI
from typing import List, Dict
from pydantic import BaseModel


# ========== Constants ==========
TOKENLIMIT = 5000
APIKEY_PATH = "./api_keys/openai/api_key.txt"
api_key = open(APIKEY_PATH).read().strip()
client = OpenAI(api_key=api_key)
SYSTEM_MESSAGE = """gpt, find the two entities (which must be people, such as the grandfather, the son, etc) in the prompt below. Replace all occurrences of the entities with Entity0 and Entity1 respectively. Give the output in this json format: {"prompt": <new prompt without Entity: <entity name>>, "A": <new ans0>, "B": <new ans1>, "C": <new ans2>} without additional text. Make sure to preserve the original label-to-entity relationship."""
MODEL_NAME = "gpt-4.1-2025-04-14"

# ========== Pydantic Model ==========
class PromptOutput(BaseModel):
    prompt: str
    A: str
    B: str
    C: str

# ========== GPT-Available Toolkit ==========

def remove_color_base64_image(base64_image_str: str) -> str:
    # Remove data URI prefix if present
    if base64_image_str.startswith("data:image"):
        base64_image_str = base64_image_str.split(",")[1]

    # Decode base64 to image
    image_data = base64.b64decode(base64_image_str)
    image = Image.open(BytesIO(image_data))

    # Convert to grayscale
    grayscale_image = image.convert("L")

    # Save to buffer
    buffer = BytesIO()
    grayscale_image.save(buffer, format="jpeg")
    buffer.seek(0)

    # Encode to base64
    grayscale_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    return f"data:image/jpeg;base64,{grayscale_base64}"


def sketch_base64_image(base64_image_str: str) -> str:
    if base64_image_str.startswith("data:image"):
        base64_image_str = base64_image_str.split(",")[1]

    image_data = base64.b64decode(base64_image_str)
    image = Image.open(BytesIO(image_data)).convert("L")  # Grayscale

    # Invert the image
    inverted = ImageOps.invert(image)

    # Apply Gaussian Blur
    blurred = inverted.filter(ImageFilter.GaussianBlur(radius=10))

    # Dodge blend (lighten effect)
    def dodge(front, back):
        result = []
        for f, b in zip(front.getdata(), back.getdata()):
            val = min(int(b * 255 / (256 - f + 1)), 255)
            result.append(val)
        blended = Image.new("L", front.size)
        blended.putdata(result)
        return blended

    sketch = dodge(image, blurred)

    # Encode back to base64
    buffer = BytesIO()
    sketch.save(buffer, format="jpeg")
    buffer.seek(0)
    sketch_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    return f"data:image/jpeg;base64,{sketch_base64}"


def compress_base64_image_to_token_limit(base64_image_str: str, token_limit: int) -> str:
    """
    Compress a base64-encoded image so that its token count is below token_limit.
    Returns a base64-encoded JPEG string with data URI prefix.
    """
    # Remove data URI prefix if present
    if base64_image_str.startswith("data:image"):
        base64_image_str = base64_image_str.split(",", 1)[1]

    # Decode base64 to image
    image_data = base64.b64decode(base64_image_str)
    image = Image.open(BytesIO(image_data))

    # Set up tiktoken encoding for OpenAI models (use cl100k_base for GPT-4/3.5)
    enc = tiktoken.get_encoding("cl100k_base")

    # Start with initial quality and size
    quality = 85
    min_quality = 20
    resize_factor = 0.9
    min_size = 32
    width, height = image.size

    while True:
        # Resize if needed
        if width > min_size and height > min_size:
            img_resized = image.resize((int(width), int(height)), Image.LANCZOS)
        else:
            img_resized = image
        # Compress to JPEG
        buffer = BytesIO()
        img_resized.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        compressed_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        data_uri = f"data:image/jpeg;base64,{compressed_base64}"
        token_count = len(enc.encode(data_uri))
        if token_count <= token_limit or (quality <= min_quality and width <= min_size and height <= min_size):
            return data_uri
        # Reduce quality first, then size
        if quality > min_quality:
            quality -= 10
        else:
            width = max(int(width * resize_factor), min_size)
            height = max(int(height * resize_factor), min_size)


# ========== Textual Tools ==========
def debias_prompt(orig_prompt: str, ans: List[str]) -> PromptOutput:
    user_prompt = orig_prompt+"\n"+"\n".join([f"{chr(ord("A")+i)}: {ans[i]}\n" for i in range(3)])
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": user_prompt}
    ]
    
    response_format = PromptOutput.schema()
    response_format["required"] = ["prompt", "choices"]
    response = client.chat.completions.parse(
        model=MODEL_NAME,
        messages=messages,
        response_format=PromptOutput,
        timeout=60
    )
    
    # Extract and parse the JSON response
    if response.choices and response.choices[0].message:
        return response.choices[0].message.parsed
    else:
        print("No response content")
        return None


# === Main driver ===

def show_images_side_by_side(original: Image.Image, grayscale: Image.Image, sketch: Image.Image, compressed: Image.Image):
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    axs[0].imshow(original)
    axs[0].set_title("Original")
    axs[0].axis("off")

    axs[1].imshow(grayscale, cmap="gray")
    axs[1].set_title("Grayscale")
    axs[1].axis("off")

    axs[2].imshow(sketch, cmap="gray")
    axs[2].set_title("Sketch")
    axs[2].axis("off")

    axs[3].imshow(compressed)
    axs[3].set_title(f"Compressed (≤{(TOKENLIMIT+999)//1000}k tokens)")
    axs[3].axis("off")

    plt.tight_layout()
    plt.show()

def process_image(image_path: str):
    base64_img = encode_image_to_base64(image_path)
    grayscale_base64 = remove_color_base64_image(base64_img)
    sketch_base64 = sketch_base64_image(base64_img)
    compressed_base64 = compress_base64_image_to_token_limit(base64_img, TOKENLIMIT)

    original = Image.open(image_path)
    grayscale = decode_base64_to_image(grayscale_base64)
    sketch = decode_base64_to_image(sketch_base64)
    compressed = decode_base64_to_image(compressed_base64)

    show_images_side_by_side(original, grayscale, sketch, compressed)


def show_compression_grid(image_path: str, token_limits: list[int]):
    """
    Display the original image and its compressed versions (for each token limit) side by side.
    Args:
        image_path: Path to the image file.
        token_limits: List of integer token limits for compression.
    """
    base64_img = encode_image_to_base64(image_path)
    original = Image.open(image_path)
    compressed_imgs = []
    labels = ["Original"]
    for limit in token_limits:
        compressed_base64 = compress_base64_image_to_token_limit(base64_img, limit)
        compressed_img = decode_base64_to_image(compressed_base64)
        compressed_imgs.append(compressed_img)
        labels.append(f"≤{(limit+999)//1000}k tokens")

    n = 1 + len(token_limits)
    fig, axs = plt.subplots(1, n, figsize=(5*n, 5))
    axs[0].imshow(original)
    axs[0].set_title("Original")
    axs[0].axis("off")
    for i, (img, label) in enumerate(zip(compressed_imgs, labels[1:]), 1):
        axs[i].imshow(img)
        axs[i].set_title(label)
        axs[i].axis("off")
    plt.tight_layout()
    plt.show()


# Tool schemas for image processing functions
TEXT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "anonymize_entity",
            "description": "If the text relates to entities which may be associated with stereotypes (e.g. the 33-year-old, the black man), this tool can reduce bias by anonymizing entity names",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False
            },
            "strict": True
        }
    }
]

IMAGE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "remove_color_base64_image",
            "description": "Convert a base64 encoded image to grayscale by removing all color information",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_id": {
                        "type": "integer",
                        "enum": [0, 1],
                        "description": "0 refers to the first image while 1 refers to the second image."
                        },
                "additionalProperties": False
                },
                "required": ["image_id"]
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "sketch_base64_image",
            "description": "Convert a base64 encoded image to a sketch effect using edge detection and dodge blending",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_id": {
                        "type": "integer",
                        "enum": [0, 1],
                        "description": "0 refers to the first image while 1 refers to the second image."
                        }
                },
                "additionalProperties": False,
                "required": ["image_id"]
            },
            "strict": True
        }
    }
]


# === Example usage ===

if __name__ == "__main__":
    image_path = "./ai_images/Age/My daughter.jpg"  # ← Replace this with the path to your image
    process_image(image_path)
    show_compression_grid(image_path, [50000, 30000, 10000, 5000, 2500])