import base64
from io import BytesIO
from PIL import Image, ImageFilter, ImageOps
import matplotlib.pyplot as plt
from utils import encode_image_to_base64, decode_base64_to_image

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
    grayscale_image.save(buffer, format="PNG")
    buffer.seek(0)

    # Encode to base64
    grayscale_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    return f"data:image/png;base64,{grayscale_base64}"


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
    sketch.save(buffer, format="PNG")
    buffer.seek(0)
    sketch_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    return f"data:image/png;base64,{sketch_base64}"


# === Main driver ===

def show_images_side_by_side(original: Image.Image, grayscale: Image.Image, sketch: Image.Image):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(original)
    axs[0].set_title("Original")
    axs[0].axis("off")

    axs[1].imshow(grayscale, cmap="gray")
    axs[1].set_title("Grayscale")
    axs[1].axis("off")

    axs[2].imshow(sketch, cmap="gray")
    axs[2].set_title("Sketch")
    axs[2].axis("off")

    plt.tight_layout()
    plt.show()

def process_image(image_path: str):
    base64_img = encode_image_to_base64(image_path)
    grayscale_base64 = remove_color_base64_image(base64_img)
    sketch_base64 = sketch_base64_image(base64_img)

    original = Image.open(image_path)
    grayscale = decode_base64_to_image(grayscale_base64)
    sketch = decode_base64_to_image(sketch_base64)

    show_images_side_by_side(original, grayscale, sketch)

# === Example usage ===

if __name__ == "__main__":
    image_path = "./ai_images/Age/My daughter.jpg"  # ← Replace this with the path to your image
    process_image(image_path)