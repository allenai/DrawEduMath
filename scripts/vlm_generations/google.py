"""

Generate text from an image using Google's Generative AI API.

@imaslarab

"""

import sys

sys.path.append("..")

import base64

import google.generativeai as genai
from dotenv import dotenv_values
from PIL import Image


class GoogleAIImageToText:
    def __init__(self):
        config = dotenv_values("../.env")
        api_key = config.get("GEMINI_API_KEY")
        # self.model = "gemini-1.5-flash"
        self.model = "gemini-1.5-pro"

        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel(self.model)

    def get_response(self, image_path, system_prompt, user_prompt=None):
        """Send a chat completion request with the image input."""

        user_prompt = user_prompt or "Describe the image."
        image = Image.open(image_path)

        response = self.client.generate_content([system_prompt + " " + user_prompt, image])

        return response.text


if __name__ == "__main__":
    image_path = "/path/to/images/0f56fb1b-c5f6-405a-b97d-5adb1bd758b8.jpeg"
    prompt = """
    You are a teacher and are given a student’s handwritten work in an image format. The handwritten work is a response 
    to a problem. Write a description for this image to explain everything in the image of the student’s handwritten work 
    in as much detail as you can so that another teacher can understand and reconstruct the math work in this image without 
    viewing the image. Focus on describing the student’s answers in the image. Your response should be a paragraph without bullet points.
    """

    anthropic_api = GoogleAIImageToText()
    response = anthropic_api.get_response(image_path, prompt)
    print(response)
