IMAGE_CAPTION = """You are a teacher and are given a student’s handwritten work in an image format. 
The handwritten work is a response to a problem. Write a description for the Student's Response provided on the right side
of each image. The description you write should explain everything in the image of the student’s handwritten work in as 
much detail as you can so that another teacher can understand and reconstruct the math work in this image without viewing
the image. The problem is provided on the left for context. If the student response is for a subproblem of a problem, 
the subproblem will be contained in a red box. Describe the student’s response on the right of the image. 
DONOT describe the problem provided on the left. Your response should be a paragraph without bullet points.
"""

DECOMPOSE_PROMPT = """You are given a detailed description of an image. The image is a student's answer to a math question. 
Decompose this detailed description into a list of short atomic descriptions, each about only one salient aspect of the image. 
Each atomic description should be a single sentence.
Output your answer in JSON.
For example, given this image description:
    'This is a natural, handwritten horizontal number line on lined paper. There are small vertical tick marks, 
    evenly spaced along the number line, denoting different integers. From left to right, each vertical tick mark 
    is labeled underneath it with the integers -12, -8, -6, -4, -2, 0, 2, 4, 6, 8, 12. The 2 is circled. There is a 
    handwritten arrow starting above 0 and ending above -6. There is a handwritten arrow starting above -6 and ending 
    above 6. There is another handwritten arrow starting above 6 and ending above 2. The students’ answer shows the 
    summation of positive and negative integers (-6, 12, -4) using a number line, with even-numbered intervals. The 
    directions of the arrows illustrate three summation steps: 0 + -6 = -6, -6 + 12 = 6, and 6 + -4 = 2. The student
    correctly annotated and circled their final answer 2. The student missed labeling 10 on the number line between 
    -12 and -8, and 8 and 12.'

Decompose the description as follows: 
    ["This is a natural image",
    "This is handwritten on lined paper.”,
    “This is a horizontal number line”,
    "There are small vertical tick marks",
    "The vertical tick marks are evenly spaced along the number line.",
    "The tick marks denote different integers.",
    "From left to right, each vertical tick mark is labeled underneath with the integers -12, -8, -6, -4, -2, 0, 2, 4, 6, 8, 12.",
    "The 2 is circled.",
    "There is a handwritten arrow starting above 0 and ending above -6.",
    "There is a handwritten arrow starting above -6 and ending above 6.",
    "There is another handwritten arrow starting above 6 and ending above 2.",
    "The student's answer shows the summation of positive and negative integers (-6, 12, -4) using a number line with even-numbered intervals.",
    "The directions of the arrows illustrate three summation steps: 0 + -6 = -6, -6 + 12 = 6, and 6 + -4 = 2.",
    "The student correctly annotated and circled their final answer 2",
    "The student missed labeling 10 on the number line between -12 and -8",
    "The student missed labeling 10 on the number line between 8 and 12."]
"""

CATEGORIZE_TIERS = """
You are given a list of short atomic descriptions, each about only one salient aspect of the image. 

Organize the list of these short atomic descriptions into different tiers which are described as:
Tier0: The descriptions in  tier 0 pertain to the quality or other generic features of the image such as background, 
type of ink used, blurriness etc. This is a non-math description of the image. If there is no tier 0 description in 
the provided list, return an empty list for this tier.
Tier1: The description in  tier 1 pertain to the the low-level description of the math content in the image.
Tier2: The description in tier 2 involves some pedagogical concepts, and is a mid-level math inference of the content in the image. 
Tier3: The tier 3 description is the high-level inference of the student's work. This tier includes information on misconception 
or error made in the math work. If these is no misconception or error in the list of descriptions, return an empty list for this tier.

If there is no description present for a tier in the provided list, return an empty list for that tier.

Output your answer in JSON.
For example, given a list of these short descriptions:
    ["This is a natural image",
    "This is handwritten on lined paper.”,
    “This is a horizontal number line”,
    "There are small vertical tick marks",
    "The vertical tick marks are evenly spaced along the number line.",
    "The tick marks denote different integers.",
    "From left to right, each vertical tick mark is labeled underneath with the integers -12, -8, -6, -4, -2, 0, 2, 4, 6, 8, 12.",
    "The 2 is circled.",
    "There is a handwritten arrow starting above 0 and ending above -6.",
    "There is a handwritten arrow starting above -6 and ending above 6.",
    "There is another handwritten arrow starting above 6 and ending above 2.",
    "The student's answer shows the summation of positive and negative integers (-6, 12, -4) using a number line with even-numbered intervals.",
    "The directions of the arrows illustrate three summation steps: 0 + -6 = -6, -6 + 12 = 6, and 6 + -4 = 2.",
    "The student correctly annotated and circled their final answer 2",
    "The student missed labeling 10 on the number line between -12 and -8",
    "The student missed labeling 10 on the number line between 8 and 12."]

Categorize into tiers as follows:
    {"Tier 0": [
        "This is a natural image",
        "This is handwritten on lined paper.",
        ],
    "Tier 1": [
        “This is a horizontal number line”,
        "There are small vertical tick marks",
        "The vertical tick marks are evenly spaced along the number line.",
        "The tick marks denote different integers.",
        "From left to right, each vertical tick mark is labeled underneath with the integers -12, -8, -6, -4, -2, 0, 2, 4, 6, 8, 12.",
        "The 2 is circled.",
        "There is a handwritten arrow starting above 0 and ending above -6.",
        "There is a handwritten arrow starting above -6 and ending above 6.",
        "There is another handwritten arrow starting above 6 and ending above 2."
        ],
    "Tier 2": [
        "The student's answer shows the summation of positive and negative integers (-6, 12, -4) using a number line with even-numbered intervals.",
        "The directions of the arrows illustrate three summation steps: 0 + -6 = -6, -6 + 12 = 6, and 6 + -4 = 2.",
        ],
    "Tier 3": [
        "The student correctly annotated and circled their final answer 2.",
        "The student missed labeling 10 on the number line between -12 and -8", 
        "The student missed labeling 10 on the number line between and 8 and 12."]
    }} 
"""