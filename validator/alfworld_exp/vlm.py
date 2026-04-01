from google import genai
from google.genai import types
import json
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import os
import base64
from typing import List, Dict, Any
import sys

def get_description(image_bytes):
    json_schema = {
      "$schema": "http://json-schema.org/draft-07/schema#",
      "type": "object",
      "properties": {
        "description": {
          "type": "string",
          "description": "Detailed description of the image."
        },
        "object_descritions": {
          "type": "array",
          "description": "List of objects in the image with their details.",
          "items": {
            "type": "object",
            "properties": {
              "object_name": {
                "type": "string",
                "description": "Name of the object."
              },
              "object_description": {
                "type": "string",
                "description": "Description of the object."
              }
            },
            "required": ["object_name", "object_description"]
          }
        }
      },
      "required": ["description", "object_descritions"]
    }
    try:
        client = genai.Client()
        response = client.models.generate_content(
                model='gemini-2.0-flash-001',
                contents=[
                    'Make a detailed description of image. Extract the properties of the objects that appear in the image. If there are multiple object of the same name, name them as object_1, object_2, ...',
                    types.Part.from_bytes(data=image_bytes, mime_type='image/png'),
                ],
                config={
                    'response_mime_type': 'application/json',
                    'response_json_schema': json_schema
                },
        )
        return response.parsed

    except Exception as e:
        # Catch any other unexpected errors.
        print(f"An unexpected error occurred: {e}")

def satisfies(image_bytes, predicate):
    json_schema = {
      "$schema": "http://json-schema.org/draft-07/schema#",
      "type": "object",
      "properties": {
        "result": {
          "type": "boolean",
          "description": "Whether the predicate satisfies the image content"
        }
      },
      "required": ["result"]
    }
    try:
        client = genai.Client()
        response = client.models.generate_content(
                model='gemini-2.0-flash-001',
                contents=[
                    f'Check whether the image content satisfies the predicate: {predicate}.',
                    types.Part.from_bytes(data=image_bytes, mime_type='image/png'),
                ],
                config={
                    'response_mime_type': 'application/json',
                    'response_json_schema': json_schema
                },
        )
        return response.parsed

    except Exception as e:
        # Catch any other unexpected errors.
        print(f"An unexpected error occurred: {e}")

def detect_object(image_bytes):
    json_schema = {
      "$schema": "https://json-schema.org/draft/2020-12/schema",
      "title": "ObjectDetectionResult",
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "object": {
            "type": "string",
            "description": "Class label of the detected object"
          },
          "bbox": {
            "type": "array",
            "description": "Bounding box in (ymin, xmin, ymax, xmax), values normalized to 0-1000.",
            "items": {
              "type": "number",
              "minimum": 0.0,
              "maximum": 1000.0
            },
            "minItems": 4,
            "maxItems": 4
          },
          "confidence": {
            "type": "number",
            "description": "Confidence score of detection between 0.0 and 1.0",
            "minimum": 0.0,
            "maximum": 1.0
          }
        },
        "required": ["object", "bbox", "confidence"]
      }
    }
    try:
        client = genai.Client()
        response = client.models.generate_content(
                model='gemini-2.5-pro-001',
                contents=[
                    f'You are an object detection system. \
                      Detect all objects in the input image and return bounding box, class label, and confidence for each detected object. \
                      Each bounding box must be formatted as: (ymin, xmin, ymax, xmax) \
                      All values must be normalized to 0-1000. \
                      left and top represent the coordinates of the upper-left corner of the bounding box. \
                      width and height represent the size of the bounding box. \
                      Available classes are AlarmClock, Apple, AppleSliced, BaseballBat, BasketBall, Book, Bowl, Box, Bread, BreadSliced, ButterKnife, \
                      CD, Candle, CellPhone, Cloth, CreditCard, Cup, DeskLamp, DishSponge, Egg, Faucet, FloorLamp, Fork, Glassbottle, HandTowel, HousePlant, \
                      Kettle, KeyChain, Knife, Ladle, Laptop, LaundryHamperLid, Lettuce, LettuceSliced, LightSwitch, Mug, Newspaper, Pan, PaperTowel, \
                      PaperTowelRoll, Pen, Pencil, PepperShaker, Pillow, Plate, Plunger, Pot, Potato, PotatoSliced, RemoteControl, SaltShaker, ScrubBrush, \
                      ShowerDoor, SoapBar, SoapBottle, Spatula, Spoon, SprayBottle, Statue, StoveKnob, TeddyBear, Television, TennisRacket, TissueBox, ToiletPaper, \
                      ToiletPaperRoll, Tomato, TomatoSliced, Towel, Vase, Watch, WateringCan, WineBottle',
                    types.Part.from_bytes(data=image_bytes, mime_type='image/png'),
                ],
                config={
                    'response_mime_type': 'application/json',
                    'response_json_schema': json_schema
                },
        )
        resp = response.parsed
        for obj in resp:
            bbox = obj["bbox"]
            obj["bbox"] = [num / 1000 for num in bbox]
        return resp

    except Exception as e:
        # Catch any other unexpected errors.
        print(f"An unexpected error occurred: {e}")


def draw_bboxes(image_path, detections, output_path=None):
    """
    Draw bounding boxes on an image.

    Args:
        image_path (str): Path to the input image.
        detections (list): List of detection dicts with format:
            {
              "object": str,
              "bbox": [left, top, width, height],  # normalized [0.0, 1.0]
              "confidence": float
            }
        output_path (str, optional): If provided, saves the image with bboxes.
    """
    # Load image
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    img_w, img_h = image.size

    for det in detections:
        obj = det["object"]
        conf = det["confidence"]
        ymin, xmin, ymax, xmax = det["bbox"]

        # Convert normalized coords to pixel values
        x0 = int(xmin * img_w)
        y0 = int(ymin * img_h)
        x1 = int(xmax * img_w)
        y1 = int(ymax * img_h)

        # Draw rectangle
        draw.rectangle([x0, y0, x1, y1], outline="red", width=1)

        # Draw label
        label = f"{obj} {conf:.2f}"
        draw.text((x0 + 5, y0 + 5), label, fill="yellow")

    # Save or show
    if output_path:
        image.save(output_path)
    else:
        image.show()

def detect_object_types(image_bytes, object_types, provider: str = 'gemini', model: str | None = None, openai_base_url: str | None = None):
    json_schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "label": {"type": "string"},
                "bbox": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 4,
                    "maxItems": 4
                },
                "confidence": {"type": "number"}
            },
            "required": ["label", "bbox", "confidence"]
        }
    }

    object_type_str = ", ".join(object_types)

    prompt = (
        f"You are an object detection system. "
        f"Detect objects [{object_type_str}] in the input image and return bounding box, class label, and confidence for each detected object. "
        f"Each bounding box must be formatted as: (ymin, xmin, ymax, xmax). "
        f"All values must be normalized to 0-1000. "
        f"ymin and xmin represent the coordinates of the upper-left corner of the bounding box. "
        f"ymax and xmax represent the coordinates of the bottom-right corner of the bounding box. "
    )

    def _scale_and_normalize(objs: List[Dict[str, Any]]):
        for obj in objs:
            bbox = obj.get("bbox", [0, 0, 0, 0])
            obj["bbox"] = [num / 1000 for num in bbox]
        return objs

    if provider == 'gemini':
        # Existing Gemini baseline
        client = genai.Client()
        response = client.models.generate_content(
            model=model or 'gemini-2.0-flash-001',
            contents=[
                prompt,
                types.Part.from_bytes(data=image_bytes, mime_type='image/png'),
            ],
            config={
                'response_mime_type': 'application/json',
                'response_json_schema': json_schema
            },
        )
        resp = response.parsed
        return _scale_and_normalize(resp)

    elif provider == 'openai':
        # OpenAI-compatible path (e.g., OpenRouter/Groq) for Llama Vision models
        # Note: Llama 3.1 is text-only; use a vision-capable Llama (e.g., a 3.2 Vision variant) with your provider.
        try:
            from openai import OpenAI
        except Exception:
            raise RuntimeError("OpenAI SDK not installed. Please `pip install openai` to use the openai provider.")

        # Configure base URL and API key (supports OpenRouter/Groq)
        base_url = openai_base_url or os.getenv('OPENAI_BASE_URL')
        api_key = os.getenv('OPENAI_API_KEY') or os.getenv('OPENROUTER_API_KEY') or os.getenv('GROQ_API_KEY')
        if not api_key:
            raise RuntimeError("Missing API key. Set OPENAI_API_KEY (or OPENROUTER_API_KEY/GROQ_API_KEY) in environment.")

        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        client = OpenAI(**client_kwargs)

        model_name = model or os.getenv('OPENAI_MODEL', 'meta-llama/llama-3.2-11b-vision-instruct')

        # Build data URL for image
        b64 = base64.b64encode(image_bytes).decode('ascii')
        data_url = f"data:image/png;base64,{b64}"

        sys_prompt = (
            "Return STRICT JSON matching this schema: "
            + json.dumps(json_schema)
            + ". No prose, no code fences."
        )

        msg = [
            {"role": "system", "content": sys_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ]

        comp = client.chat.completions.create(
            model=model_name,
            messages=msg,
            temperature=0.0,
            max_tokens=1024,
        )
        text = comp.choices[0].message.content if comp and comp.choices else "[]"
        # Try to extract JSON if wrapped
        if not text.strip().startswith("["):
            start = text.find("[")
            end = text.rfind("]")
            if start != -1 and end != -1 and end > start:
                text = text[start:end+1]
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = []
        return _scale_and_normalize(parsed)
    else:
        raise ValueError(f"Unknown provider '{provider}'. Use 'gemini' or 'openai'.")


# --- Example Usage ---
if __name__ == "__main__":
    with open('image.png', 'rb') as f:
        image_bytes = f.read()
    #print(get_description(image_bytes))
    #print(satisfies(image_bytes, "It is possible to pick up the keyboard."))
    result = detect_object(image_bytes)
    draw_bboxes('image.png', result)