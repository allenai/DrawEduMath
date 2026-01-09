GENERATE_ANSWER_PROMPT = """You are given an image of a student's handwritten work in response to a math problem.
The student's work is shown on the right side of the image, and the problem is displayed on the left side for context.
Your task is to answer a question based solely on the visual content of the student's handwritten work,
which is present on the right side of the image. Your answer should be clear and concise, and directly relate to the image presented
on the right side of the given image.
For example, given the question:
"What is the equation shown in the image?"
Generate your answer as: "3x + 2 = 8"
"""

JUDGE_PROMPT_TEMPLATE = """
Given the following inputs:
Question: {question}
Answer 1 (Ground Truth): {teacher_a}
Answer 2 (Model Output): {model_a}

Your task is to rate the quality of Answer 2, using Answer 1 as the ground truth for a perfect response. Use the following 4-point scale:


**4: Semantically Identical**
- Answer 2 conveys the exact same meaning as Answer 1.
- It is fully correct and complete. Wording differences are acceptable.

**3: Different but valid**
- Answer 2 is factually correct and answers the question, but omits some details, nuance, or key points found in Answer 1, or adds some additional, plausible insight.
- The core idea is right, but the substance is partial.

**2: Factually incorrect in an important way**
- Answer 2 attempts to answer the question, but is factually incorrect in a significant way.
- Or, it mixes correct information with fabricated/hallucinated details not supported by the question or Answer 1.

**1: Irrelevant or Wrong**
- Answer 2 completely fails to answer the question, is on a different topic, or is nonsensical.


Provide both the Likert rating followed by a brief explanation for your choice. Format the output as a valid parsable JSON like: {{"rating": 1-4, "reason": "Your brief justification here."}}"""
