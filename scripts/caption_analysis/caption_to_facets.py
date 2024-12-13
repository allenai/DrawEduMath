"""

Script to generate texts from a given text using the Together api and Mixtral 7X22B model.

@Sami Baral

"""

import sys

sys.path.append("..")

from data.caption_prompt import prompt_v1
from dotenv import dotenv_values
from together import Together


class TogetherTextToText:
    def __init__(self):
        config = dotenv_values(".env")
        api_key = config.get("TOGETHER_API_KEY")
        self.model = "mistralai/Mixtral-8x22B-Instruct-v0.1"
        self.client = Together(api_key=api_key)

    def get_response(self, system_prompt, user_prompt):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        return response.choices[0].message.content


if __name__ == "__main__":
    together_text = TogetherTextToText()

    system_prompt = prompt_v1.DECOMPOSE_PROMPT
    caption = """
    This is a digital, typed and handwritten image. There is a graph. One point on the graph is shaded in with a circe. 
    The student has drawn a downward-sloping line through the point. The student has drawn a slope triangle beneath the 
    line. The vertical part of the slope triangle has a label of -12. The horizontal part of the slope triangle has a 
    label of 6. This shows the line has a slope of -2.
    """
    user_prompt = f"Decompose this caption: {caption}"

    response = together_text.get_response(system_prompt, user_prompt)
    print("Output: ")
    print(response)
