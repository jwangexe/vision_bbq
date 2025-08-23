import base64
from io import BytesIO
from PIL import Image, ImageFilter, ImageOps
import matplotlib.pyplot as plt
from utils import encode_image_to_base64, decode_base64_to_image
from openai import OpenAI
from typing import List, Dict
from pydantic import BaseModel
import os


# ========== Constants ==========
TOKENLIMIT = 5000
API_KEY = os.getenv("API_KEY", "ollama")
client = OpenAI(api_key=API_KEY)
SYSTEM_MESSAGE = """Find the two people mentioned in the prompt. Replace all their names with "Entity0" and "Entity1" in the prompt and answers. Return the result as JSON:
{
  "prompt": <modified prompt>,
  "A": <modified ans0>,
  "B": <modified ans1>,
  "C": <modified ans2>
}
Keep the original answer-to-entity mapping. Do not add any extra text."""
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


def resize_base64_jpg(base64_str: str, max_size: int = 512) -> str:
    # Decode base64 to bytes
    if base64_str.startswith("data:image"):
        base64_str = base64_str.split(",")[1]
    image_data = base64.b64decode(base64_str)
    
    # Open image from bytes
    with Image.open(BytesIO(image_data)) as img:
        # Calculate new size keeping aspect ratio
        width, height = img.size
        if width > height:
            new_width = max_size
            new_height = int((max_size / width) * height)
        else:
            new_height = max_size
            new_width = int((max_size / height) * width)
        
        # Resize image
        resized_img = img.resize((new_width, new_height), Image.LANCZOS)
        
        # Save to bytes buffer as JPEG
        buffered = BytesIO()
        resized_img.save(buffered, format="JPEG")
        
        # Encode resized image to base64
        resized_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
    return f"data:image/jpeg;base64,{resized_base64}"


# ========== Textual Tools ==========
def debias_prompt(orig_prompt: str, ans: List[str]) -> PromptOutput:
    user_prompt = orig_prompt+"\n"+"\n".join([f"{'ABC'[i]}: {ans[i]}\n" for i in range(3)])
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
    compressed_base64 = resize_base64_jpg(base64_img)

    original = Image.open(image_path)
    grayscale = decode_base64_to_image(grayscale_base64)
    sketch = decode_base64_to_image(sketch_base64)
    compressed = decode_base64_to_image(compressed_base64)

    show_images_side_by_side(original, grayscale, sketch, compressed)



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
            "name": "grayscale",
            "description": "Convert a base64 encoded image to grayscale by removing all color information",
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
    },
    {
        "type": "function",
        "function": {
            "name": "sketch",
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