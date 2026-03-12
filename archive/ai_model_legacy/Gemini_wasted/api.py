from google import genai
from google import genai
from google.genai import types
import pathlib
client = genai.Client()

image_data = pathlib.Path("/media/qichang/Data/research/Gas_price/data/bright/4.png").read_bytes()

response = client.models.generate_content(
    model="gemini-3-flash-preview",
        contents=[
        types.Part.from_bytes(data=image_data, mime_type="image/jpeg"),
        "Describe what you see in this image",
    ]
)
# add time calculation mechanism
print(response.text)