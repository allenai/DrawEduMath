"""

Generate text from an image using Anthropic's Generative AI API.

@imaslarab

"""

import sys

sys.path.append("..")

import base64

import anthropic
from dotenv import dotenv_values
from PIL import Image


class AnthropicImageToText:
    def __init__(self):
        config = dotenv_values("../.env")
        api_key = config.get("ANTHROPIC_API_KEY")
        self.model = "claude-3-5-sonnet-20240620"

        self.client = anthropic.Anthropic(api_key=api_key)

    def encode_image(self, image_path):
        """Encode the image in base64 format."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def get_image_format(self, image_path):
        image = Image.open(image_path)
        image_format = ("image/" + image.format).lower()

        return image_format

    def get_response(self, image_path, system_prompt, user_prompt=None):
        """Send a chat completion request with the image input."""

        user_prompt = user_prompt or "Describe the image."
        encoded_image = self.encode_image(image_path)
        image_media_type = self.get_image_format(image_path)

        response = self.client.messages.create(
            model=self.model,
            max_tokens=256,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": image_media_type,
                                "data": encoded_image,
                            },
                        },
                    ],
                }
            ],
        )

        return response.content[0].text


if __name__ == "__main__":
    image_path = "/path/to/images/0f56fb1b-c5f6-405a-b97d-5adb1bd758b8.jpeg"
    prompt = """
    You are a teacher and are given a student’s handwritten work in an image format. The handwritten work is a response 
    to a problem. Write a description for this image to explain everything in the image of the student’s handwritten work 
    in as much detail as you can so that another teacher can understand and reconstruct the math work in this image without 
    viewing the image. Focus on describing the student’s answers in the image. Your response should be a paragraph without bullet points.
    """

    anthropic_api = AnthropicImageToText()
    response = anthropic_api.get_response(image_path, prompt)
    print(response)
