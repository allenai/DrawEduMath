# Extract Facets --> Generate + Categorize QA Pairs into Tiers
DECOMPOSE_PROMPT = """
You are given a detailed description of an image. The image is a student's answer to a math question. 
Decompose this description into atomic descriptions, each about only one salient aspect of the image. 
Output your answer in JSON.
For example give this image description:
    'This is a natural, handwritten horizontal number line on lined paper. There are small vertical tick marks, 
    evenly spaced along the number line, denoting different integers. From left to right, each vertical tick mark 
    is labeled underneath it with the integers -12, -8, -6, -4, -2, 0, 2, 4, 6, 8, 12. The 2 is circled. There is a 
    handwritten arrow starting above 0 and ending above -6. There is a handwritten arrow starting above -6 and ending 
    above 6. There is another handwritten arrow starting above 6 and ending above 2. The students’ answer shows the 
    summation of positive and negative integers (-6, 12, -4) using a number line, with even-numbered intervals. The 
    directions of the arrows illustrate three summation steps: 0 + -6 = -6, -6 + 12 = 6, and 6 + -4 = 2. '

Generate: 
    ["This is a natural image’, handwritten on lined paper.”,
    “This is a horizontal number line”,
    "There are small vertical tick marks, evenly spaced along the number line.",
    "The tick marks denote different integers.",
    "From left to right, each vertical tick mark is labeled underneath with the integers -12, -8, -6, -4, -2, 0, 2, 4, 6, 8, 12.",
    "The 2 is circled.",
    "There is a handwritten arrow starting above 0 and ending above -6.",
    "There is a handwritten arrow starting above -6 and ending above 6.",
    "There is another handwritten arrow starting above 6 and ending above 2.",
    "The student's answer shows the summation of positive and negative integers (-6, 12, -4) using a number line with even-numbered intervals.",
    "The directions of the arrows illustrate three summation steps: 0 + -6 = -6, -6 + 12 = 6, and 6 + -4 = 2.",]
"""

QA_TIER0_PROMPT = """You're given a list of short descriptions about a specific aspect of an image. 
The image is a student's answer to a math question.
Using items in this list write a question and answer pair, where the question is relevant to each description and 
a person should be able to answer the question using information from the description. The question answer pairs
should be about the quality or other generic features of the image such as background, type of ink used, etc.
This question and answer pair should be a non-math-related description and should not include any math inference.
If there are no question and answer pairs related to this in the descriptions, return an empty string. Output your answer in JSON.

For example, given a list of these descriptions:
    [“This is a natural image’, handwritten on lined paper.”,
    “This is a horizontal number line”,
    "There are small vertical tick marks, evenly spaced along the number line.",
    "The tick marks denote different integers.",
    "From left to right, each vertical tick mark is labeled underneath with the integers -12, -8, -6, -4, -2, 0, 2, 4, 6, 8, 12.",
    "The 2 is circled.",
    "There is a handwritten arrow starting above 0 and ending above -6.",
    "There is a handwritten arrow starting above -6 and ending above 6.",
    "There is another handwritten arrow starting above 6 and ending above 2.",
    "The student's answer shows the summation of positive and negative integers (-6, 12, -4) using a number line with even-numbered intervals.",
    "The directions of the arrows illustrate three summation steps: 0 + -6 = -6, -6 + 12 = 6, and 6 + -4 = 2.",]

Generate this question answer pairs:
[{
    "question": "What type of image is this?",
    "answer": "natural image, handwritten"},
{   "question": "What kind of paper is used in the image?",
    "answer": "Lined paper"
}]
"""

QA_TIER1_PROMPT = """You're given a list of short descriptions about a specific aspect of an image. 
The image is a student's answer to a math question.
Using items in this list write a question and answer pair, where the question is relevant to each description and 
a person should be able to answer the question using information from the description. The question answer pairs
should be the low-level description of the math content in the image.
If there are no question answer pairs related to this in the descriptions, return an empty string. Output your answer in JSON.

For example, given a list of these descriptions:
    [“This is a natural image’, handwritten on lined paper.”,
    “This is a horizontal number line”,
    "There are small vertical tick marks, evenly spaced along the number line.",
    "The tick marks denote different integers.",
    "From left to right, each vertical tick mark is labeled underneath with the integers -12, -8, -6, -4, -2, 0, 2, 4, 6, 8, 12.",
    "The 2 is circled.",
    "There is a handwritten arrow starting above 0 and ending above -6.",
    "There is a handwritten arrow starting above -6 and ending above 6.",
    "There is another handwritten arrow starting above 6 and ending above 2.",
    "The student's answer shows the summation of positive and negative integers (-6, 12, -4) using a number line with even-numbered intervals.",
    "The directions of the arrows illustrate three summation steps: 0 + -6 = -6, -6 + 12 = 6, and 6 + -4 = 2.",]

Generate this question answer pairs:
[{
    "question": "What is drawn on the lined paper?",
    "answer": "horizontal number line."},
    {"question": "How are the tick marks along the number line spaced?",
    "answer": "evenly spaced."},
    {"question": "What do the tick marks on the number line represent?",
    "answer": "different integers."},
    {"question": "Which integers are labeled on the number line from left to right?",
    "answer": "-12, -8, -6, -4, -2, 0, 2, 4, 6, 8, and 12."},
    {"question": "Which number is circled on the number line?",
    "answer": "2 is circled."},
    {"question": "Where does the handwritten arrow start and end?",
    "answer": "starts above 0 and ends above -6, starts above -6 and ends above 6, starts above 6 and ends above 2."
}],
"""

QA_TIER2_PROMPT = """You're given a list of short descriptions about a specific aspect of an image. 
The image is a student's answer to a math question.
Using items in this list write a question and answer pair, where the question is relevant to each description and 
a person should be able to answer the question using information from the description. The question answer pairs
should include pedagogical concepts, and is a mid-level math inference of the content in the image.
If there are no question answer pairs related to this in the descriptions, return an empty string. Output your answer in JSON.

For example, given a list of these descriptions:
    [“This is a natural image’, handwritten on lined paper.”,
    “This is a horizontal number line”,
    "There are small vertical tick marks, evenly spaced along the number line.",
    "The tick marks denote different integers.",
    "From left to right, each vertical tick mark is labeled underneath with the integers -12, -8, -6, -4, -2, 0, 2, 4, 6, 8, 12.",
    "The 2 is circled.",
    "There is a handwritten arrow starting above 0 and ending above -6.",
    "There is a handwritten arrow starting above -6 and ending above 6.",
    "There is another handwritten arrow starting above 6 and ending above 2.",
    "The student's answer shows the summation of positive and negative integers (-6, 12, -4) using a number line with even-numbered intervals.",
    "The directions of the arrows illustrate three summation steps: 0 + -6 = -6, -6 + 12 = 6, and 6 + -4 = 2.",]

Generate this question answer pair:
[{
    "question": "What mathematical concept does the student's work illustrate?",
    "answer": "The student's work illustrates the summation of positive and negative integers using a number line."},
    {"question": "How does the student use arrows to show their work?",
    "answer": "The arrows illustrate three summation steps: 0 + -6 = -6, -6 + 12 =  6, and 6 + -4 = 2."
}]
"""

QA_TIER3_PROMPT = """You're given a list of short descriptions about a specific aspect of an image. 
The image is a student's answer to a math question.
Using items in this list write a question and answer pair, where the question is relevant to each description and 
a person should be able to answer the question using information from the description. The question answer pairs
should be the high-level inference of the student's work. This includes information on misconception or error made 
in the math work. 
If there are no question and answer pairs related to this in the descriptions, return an empty string. Output your answer in JSON.

For example, given a list of these descriptions:
    [“This is a natural image’, handwritten on lined paper.”,
    “This is a horizontal number line”,
    "There are small vertical tick marks, evenly spaced along the number line.",
    "The tick marks denote different integers.",
    "From left to right, each vertical tick mark is labeled underneath with the integers -12, -8, -6, -4, -2, 0, 2, 4, 6, 8, 12.",
    "The 2 is circled.",
    "There is a handwritten arrow starting above 0 and ending above -6.",
    "There is a handwritten arrow starting above -6 and ending above 6.",
    "There is another handwritten arrow starting above 6 and ending above 2.",
    "The student's answer shows the summation of positive and negative integers (-6, 12, -4) using a number line with even-numbered intervals.",
    "The directions of the arrows illustrate three summation steps: 0 + -6 = -6, -6 + 12 = 6, and 6 + -4 = 2.",]

Generate this question and answer pairs:
[]
"""

