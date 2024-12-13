# Decompose + Generate QA pair 
DECOMPOSE_PROMPT = """
    You're given a long description of an image. Decompose this description into atomic descriptions, 
    each about only one salient aspect of the image. Output your answer in JSON. 
    
    For example, given this caption :  
    'This is a blurry photo of a number line that starts from -1 to +1. This has 10 evenly spaced tick marks. 
    And the student drew an arrow starting from 0 to 1. ', 
    Generate :
    [
        'This is a blurry photo',
        'The photo is of a number line.', 
        'The number line starts from -1 and goes to +1',  
        'The number line has 10 evenly-spaced tick marks',  
        'The student drew an arrow starting from 0 to 1'
    ]
    """


QA_PROMPT = """
    You're given a list of short descriptions about a specific aspect of an image. Rewrite items in 
    this list as a question and answer pair, where the question is relevant to each description and 
    a person should be able to answer the question using information from the description. 
    For example, given these descriptions:
    [
            'This is a blurry photo',
            'The photo is of a number line.', 
            'The number line starts from -1 and goes to +1',  
            'The number line has 10 evenly-spaced tick marks',  
            'The student drew an arrow starting from 0 to 1'
    ],
    generate:
    [{
        'question': What is the quality of the photo?,
        'answer': 'blurry'
    }, {
        'question': 'What math strategy is shown in this photo?',
        'answer': 'a number line'
    },{
        'question': 'What is the range of the number line?',
        'answer': '-1 to +1.'
    },{
        'question': 'How many tick marks are on the number line?',
        'answer': '10'
    },{
        'question': 'How are the tick marks spaced?',
        'answer': 'evenly'
    },{
        'question': 'Where does the student-drawn arrow start?',
        'answer': '0'
    },{
        'question': 'Where does the student-drawn arrow end?',
        'answer': '1'
    }]"""