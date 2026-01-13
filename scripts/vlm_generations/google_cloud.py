"""

Provides the GoogleCloudImageToText class, which is used to send requests to custom models hosted on Google Cloud.

@Nathan Anderson

"""

import base64
import os
import subprocess

import dotenv
from openai import OpenAI
from PIL import Image

import data.vqa_prompt.prompt_qa as prompt_qa

dotenv.load_dotenv()

PROJECT_ID = os.environ["GCLOUD_PROJECT_ID"]
ENDPOINT_REGION = os.environ["GCLOUD_ENDPOINT_REGION"]
ENDPOINT_ID = os.environ["GCLOUD_ENDPOINT_ID"]

# The the path of the binary of your Google Cloud install.
# You can find it using "where gcloud" (Windows) or "whereis gcloud" (Linux/Mac).
# Required for getting the access token.
GCLOUD_INSTALL_PATH = os.environ["GCLOUD_INSTALL_LOCATION"]

api_endpoint = f"{ENDPOINT_REGION}-aiplatform.googleapis.com"
endpoint = f"projects/{PROJECT_ID}/locations/{ENDPOINT_REGION}/endpoints/{ENDPOINT_ID}"

API_ENDPOINT_URL = f"https://{ENDPOINT_REGION}-aiplatform.googleapis.com/v1beta1/{endpoint}"


def encode_image(image_path):
    """Encode the image in base64 format."""
    image = Image.open(image_path).resize((150, 150))
    return base64.b64encode(image.tobytes()).decode("utf-8")


# Get the access token for Google Cloud.
# This requires you to have gcloud installed on your machine, unfortunately there is no way around this as far as I can tell.
# Since access tokens have a limited lifespan and the benchmark can take a few days to run, it's safest to refetch the access token every time we send a request.
def get_access_token() -> str:
    return subprocess.check_output([GCLOUD_INSTALL_PATH, "auth", "print-access-token"]).decode("utf-8").strip()


class GoogleCloudImageToText:
    def __init__(self, _): ...

    def encode_image(self, image_path):
        """Encode the image in base64 format."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def get_response(self, image_url, system_prompt, user_prompt=None):
        """Send a chat completion request with the image input."""

        user_prompt = user_prompt or "Describe the image."

        client = OpenAI(base_url=API_ENDPOINT_URL, api_key=get_access_token())

        response = client.chat.completions.create(
            model="",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                },
            ],
            max_tokens=200,
        )

        return response.choices[0].message.content


if __name__ == "__main__":
    image_path = "/path/to/image/to/run/on"
    image_url = "URL to image to fetch the image from"
    # system_prompt = """
    # You are a teacher and are given a student’s handwritten work in an image format. The handwritten work is a response
    # to a problem. Write a description for this image to explain everything in the image of the student’s handwritten work
    # in as much detail as you can so that another teacher can understand and reconstruct the math work in this image without
    # viewing the image. Focus on describing the student’s answers in the image. Your response should be a paragraph without bullet points.
    # """
    system_prompt = prompt_qa.GENERATE_ANSWER_PROMPT
    user_prompt = "Did the students label the number line correctly?"
    # user_prompt = "Did students use arrows to indicate the direction of movement on the number line?"

    gemma3_api = GoogleCloudImageToText("")
    response = gemma3_api.get_response(image_url, system_prompt, user_prompt)
    print(response)
