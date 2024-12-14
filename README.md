# DrawEduMath

DrawEduMath is a benchmark dataset of 2,030 annotated K-12 student math work images with dense captions and 11,661 QA pairs, all written by real-world teachers. 

### Using VLMs to answer questions about student math work

Run example scripts at:
```
|-- scripts/
    |-- vlm_generations/
        |-- openai.py
        |-- anthropic.py
        |-- google.py
```

Our prompts are stored in `data/qa_prompt/`

### Using LMs to analyze image captions

Run example scripts at:
```
|-- scripts/
    |-- caption_analysis/
        |-- caption_to_facets.py
```

Our prompts are stored in `data/caption_prompt/`


