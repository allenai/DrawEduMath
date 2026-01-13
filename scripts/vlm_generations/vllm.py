"""

Provides the VLLMImageToText class, which is used to send requests to models hosted via VLLM.

@Nathan Anderson

"""

import base64
from openai import OpenAI


# For usage with models hosted locally using VLLM.
class VLLMImageToText:
    def __init__(self, model):
        endpoint = "http://localhost:8000/v1"  # default vllm endpoint

        self.client = OpenAI(base_url=endpoint, api_key="EMPTY")
        self.model = model

    def encode_image(self, image_path):
        """Encode the image in base64 format."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def get_response(self, image_path, system_prompt, user_prompt=None):
        """Send a chat completion request with the image input."""

        user_prompt = user_prompt or "Describe the image."
        encoded_image = self.encode_image(image_path)

        response = self.client.chat.completions.create(
            model=self.model,  # Use the appropriate model supporting image input
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}},
                    ],
                },
            ],
            max_tokens=200,
        )

        return response.choices[0].message.content


if __name__ == "__main__":
    image_path = "/path/to/image/to/run/on"
    prompt = """
    You are a teacher and are given a student’s handwritten work in an image format. The handwritten work is a response
    to a problem. Write a description for this image to explain everything in the image of the student’s handwritten work
    in as much detail as you can so that another teacher can understand and reconstruct the math work in this image without
    viewing the image. Focus on describing the student’s answers in the image. Your response should be a paragraph without bullet points.
    """

    openai_api = VLLMImageToText("microsoft/Phi-4-multimodal-instruct")
    response = openai_api.get_response(image_path, prompt)
    print(response)
