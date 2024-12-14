# DrawEduMath

DrawEduMath is a benchmark dataset of 2,030 annotated K-12 student math work images with dense captions and 11,661 QA pairs, all written by real-world teachers. 

For more on the project, visit our website at [https://drawedumath.org](https://drawedumath.org).

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


### Citation

```
@inproceedings{baral2024drawedumath,
  author    = {Baral, Sami and Li, Lucy and Knight, Ryan and Ng, Alice and Soldainin, Luca and Heffernan, Neil and Lo, Kyle},
  title     = {DrawEduMath: Evaluating Vision Language Models with Expert-Annotated Students’ Hand-Drawn Math Images},
  booktitle = {The 4th Workshop on Mathematical Reasoning and AI at NeurIPS'24},
  year      = {2024}
}
```