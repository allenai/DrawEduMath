# Decompose + Generate QA pair 
DECOMPOSE_PROMPT = """
    You're given a long description of an image. Decompose this description into atomic descriptions, 
    each about only one salient aspect of the image. The atomic descriptions are short sentences or clauses extracted, 
    but not inferred, from the given description.
    Output your answer in JSON. 
    
    For example, given this caption :  
    'This is a hand-drawn area model on vertically lined paper. This area model has three columns and two rows. 
    Above the first column is a three, above the second column is a three, above the third column is a two. 
    To the left of the first row is a one, to the left of the second row is a 20. In the box connecting one and three,
    there's a 300. In the box connecting the second one and three, there's a 30. In the box connecting one and two,
    there's a 2. In the box connecting 20 and 3, there's a 6,000. In the box connecting 20 and the 2nd 3, 
    there's a 600. In the box connecting 20 and 2, there's a 40. On the side, the student writes 33.2 with a 664.0 beneath it. 
    There's a line indicating the sum of the 33.2 and the 664.0, indicating that the sum is 697.2.', 
    Generate :
    [
        "This is a hand-drawn area model on vertically lined paper.",
        "This area model has three columns and two rows. ",
        "Above the first column is a three, above the second column is a three, above the third column is a two.",
        "To the left of the first row is a one, to the left of the second row is a 20.",
        "In the box connecting one and three, there's a 300.",
        "In the box connecting the second one and three, there's a 30.",
        "In the box connecting one and two, there's a 2.",
        "In the box connecting 20 and 3, there's a 6,000.",
        "In the box connecting 20 and the 2nd 3, there's a 600.",
        "In the box connecting 20 and 2, there's a 40.",
        "On the side, the student writes 33.2 with a 664.0 beneath it.",
        "There's a line indicating the sum of the 33.2 and the 664.0, indicating that the sum is 697.2.",
        "There are three columns and two rows.",
        "The content in the image is hand-drawn.",
        "The drawing is on vertically lined paper.",
        "This is an area model.",
        "There's a line indicating the sum of 33.2 and 664.0 is 697.2."
    ]
    """


QA_PROMPT = """
    You're given a list of short descriptions about a specific aspect of an image. Rewrite items in 
    this list as a question and answer pair, where the question is relevant to each description and 
    a person should be able to answer the question using information from the description. 
    Output your answer in JSON.
    For example, given these descriptions:
    [
        "This is a hand-drawn area model on vertically lined paper.",
        "This area model has three columns and two rows. ",
        "Above the first column is a three, above the second column is a three, above the third column is a two.",
        "To the left of the first row is a one, to the left of the second row is a 20.",
        "In the box connecting one and three, there's a 300.",
        "In the box connecting the second one and three, there's a 30.",
        "In the box connecting one and two, there's a 2.",
        "In the box connecting 20 and 3, there's a 6,000.",
        "In the box connecting 20 and the 2nd 3, there's a 600.",
        "In the box connecting 20 and 2, there's a 40.",
        "On the side, the student writes 33.2 with a 664.0 beneath it.",
        "There's a line indicating the sum of the 33.2 and the 664.0, indicating that the sum is 697.2.",
        "There are three columns and two rows.",
        "The content in the image is hand-drawn.",
        "The drawing is on vertically lined paper.",
        "This is an area model.",
        "There's a line indicating the sum of 33.2 and 664.0 is 697.2."
    ],

    generate:
    [{
        "question": "What number is above the first column?",
        "answer": "Three",
    }, {
        "question": "What number is above the first column?", 
        "answer": "Three",
    }, {
        "question": "What number is above the second column?",
        "answer": "Three",
    }, {
        "question": "What number is above the third column?",
        "answer": "Two",
    }, {
        "question": "What number is to the left of the first row?", One.
        "answer": "One",
    }, {
        "question": "What number is to the left of the second row?",
        "answer": "20",
    }, {
        "question": "What number is in the box connecting one and three?",
        "answer": "30",
    }, {
        "question": "What number is in the box connecting one and two?", 
        "answer": "2",
    }, {
        "question": "What number is in the box connecting 20 and 3?", 
        "answer": "6,000",
    }, {
        "question": "What number is in the box connecting 20 and the 2nd 3?",
        "answer": "600",
    }, {
        "question": "What number is in the box connecting 20 and 2?",
        "answer": "40",
    }, { 
        "question": "What number is written beneath 33.2?",
        "answer": "664.0",
    }, {
        "question": "How many columns and rows are there?",
        "answer": "Three columns and two rows",
    }, {
        "question": "Is the content in the image hand-drawn?",
        "answer": "Yes",
    }, {
        "question": "What kind of paper is in the drawing on?",
        "answer": "Vertically lined paper",
    }, {
        "question": "What kind of model is this?",
        "answer": "An area model",
    }, {
        "question": "What does the line indicate?",
        "answer": "The sum of 33.2 and 664.0 is 697.2"
    }]"""

GENERATE_ANSWER_PROMPT = """
You are given an image of a student’s handwritten work in response to a math problem. 
The student's work is shown on the right side of the image, and the problem is displayed on the left side for context.
Your task is to answer a question based solely on the visual content of the student’s handwritten work,
which is present on the right side of the image. Your answer should be clear and concise, and directly relate to the image presented
on the right side of the given image. 
For example, given the question: 
"What is the equation shown in the image?"
Generate your answer as: "3x + 2 = 8"
"""


EVALUATE_ANSWER_PROMPT = """Given, Question: {question}
Answer 1: {teacher_a}
Answer 2: {model_a}

Rate the level of similarity between these two answers with respect to how well they answer this question. The Likert rating options are:
4. Basically the same answer
3. Similar but not same answer
2. Neither similar nor different
1. Quite different answers

Provide both the Likert rating followed with an explanation as to why they are similar. Format the output as a valid parsable JSON like:
{"rating": 3, "reason": "Because..."}"""

# GENERATE_ANSWER_PROMPT = """
# You are given an image of a student’s handwritten work in response to a math problem. 
# The student's work is shown on the right side of the image, and the problem is displayed on the left side for context.
# Your task is to answer a series of questions based solely on the visual content of the student’s handwritten work,
# which is present on the right side of the image.
# For each question, provide a clear and concise answer that directly relates to the visual details present in the student's work. 
# Output your answer in JSON as: 
# [
# {'question': 'What is the sum of 1 and 2?', 
# 'answer': '3'},
# ]
# """

# GENERATE_ANSWER_PROMPT = """
# You are given an image of a student’s handwritten work in response to a math problem. 
# The student's work is shown on the right side of the image, and the problem is displayed on the left side for context.
# Your task is to answer a question based solely on the visual content of the student’s handwritten work,
# which is present on the right side of the image. You answer should be clear and concise, and directly relate to the image presented
# on the right side of the given image. Your answer should be in JSON.
# For example, given the question: 
# "What is the sum of 1 and 2?"
# Generate your answer as: 
# {'question': 'What is the sum of 1 and 2?', 
# 'answer': '3'}
# """