# DrawEduMath VLM Benchmark Pipeline

Comprehensive pipeline for evaluating Vision-Language Models on the DrawEduMath benchmark (56,054 QA pairs of student handwritten math work).

## Overview

This pipeline provides a complete workflow for:
- Generating model responses from 14+ state-of-the-art VLMs
- Triple-judge evaluation with ensemble voting
- Computing evaluation metrics (BERTScore, ROUGE-L)
- Comprehensive scoring by QA source (teacher vs synthetic)

## Setup

### Install Dependencies

Install all required dependencies:

```bash
cd scripts/pipeline
pip install -r requirements.txt
```

### Configure API Keys

Create `.env` file in the project root:

```bash
ANTHROPIC_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here
TOGETHER_API_KEY=your_key_here
```

### Download Dataset

Download the DrawEduMath dataset from HuggingFace:

**Dataset:** https://huggingface.co/datasets/allenai/DrawEduMath

Extract images to:
```
data/AllImages/Resized_Merged_Problem_Images/
```

The dataset contains:
- 2,030 PNG images of student math work
- 56,054 QA pairs
- Size: ~2GB

## Pipeline

### 1. Generate Model Responses

**Available Models:**

| Provider | Model Key | API Model Name |
|----------|-----------|----------------|
| Anthropic | `claude-3.7-sonnet` | claude-3-7-sonnet-20250219 |
| Anthropic | `claude-sonnet-4` | claude-sonnet-4-20250514 |
| Anthropic | `claude-sonnet-4.5` | claude-sonnet-4-5-20250929 |
| Anthropic | `claude-opus-4.5` | claude-opus-4-5 |
| Google | `gemini-2.5-pro` | gemini-2.5-pro |
| Google | `gemini-pro-2.5-preview` | gemini-2.5-pro-preview |
| Google | `gemini-2.0-flash` | gemini-2.0-flash |
| Google | `gemini-3-pro-preview` | gemini-3-pro-preview |
| OpenAI | `gpt-4.1` | gpt-4.1-2025-04-14 |
| OpenAI | `gpt-5` | gpt-5 |
| OpenAI | `gpt-4.5-preview` | gpt-4.5-preview-2025-02-27 |
| OpenAI | `o4-mini` | o4-mini-2025-04-16 |
| OpenAI | `gpt-5.2` | gpt-5.2-2025-12-11 |
| Together | `llama-4-scout` | meta-llama/Llama-4-Scout-17B-16E-Instruct |

**Configure and Run:**

Edit `SELECTED_MODEL` in the generation script, then run:

```bash
cd scripts/pipeline/generation

# Anthropic models
python generate_anthropic.py

# Google models
python generate_google.py

# OpenAI models
python generate_openai.py

# Together AI models
python generate_together.py
```

**Output:** Creates `output/{model_name}.csv` with `Model Answer` column populated.

**Settings:**
- `RATE_LIMIT = 150` - Requests per minute
- `SLEEP_TIME = 60` - Sleep duration when limit reached
- `SAVE_INTERVAL = 10` - Checkpoint frequency

### 2. Judge Model Responses

Run three judge models to evaluate responses on a **1-4 scale**:

```bash
cd scripts/pipeline/judges

# Run each judge
python judge_claude.py ../../output/your_model.csv
python judge_gemini.py ../../output/your_model.csv
python judge_gpt4o.py ../../output/your_model.csv
```

**Judge Rating Scale:**
- **4:** Semantically identical to reference answer
- **3:** Different but valid approach/explanation
- **2:** Factually incorrect in important way
- **1:** Irrelevant or completely wrong

**Output:** Creates batch files in `output/{judge}_judge/{model}/{timestamp}/` with columns:
- `Judge_Rating` (1-4, or -1 for missing responses)
- `Judge_Reason` (explanation)

### 3. Merge Judge Ratings

After all three judges complete, merge ratings for each judge:

```bash
cd scripts/pipeline/judges

python merge_judge.py ../../output/your_model.csv claude
python merge_judge.py ../../output/your_model.csv gemini
python merge_judge.py ../../output/your_model.csv openai
```

**Output:** Updates main CSV with judge columns:
- `Claude_Judge_Rating` & `Claude_Judge_Reason`
- `Gemini_Judge_Rating` & `Gemini_Judge_Reason`
- `Openai_Judge_Rating` & `Openai_Judge_Reason`

### 4. Evaluate Results

Calculate ensemble ratings and final metrics:

```bash
cd scripts/pipeline/evaluation

python run_evaluation.py ../../output/your_model.csv
```

**Output:**
- Adds `Ensemble_Judge_Rating` column (mode/most common rating across judges)
- Optionally computes BERTScore F1 and ROUGE-L (toggle in script)

Then, print final scores:

```bash
python print_scores.py
```

**Output:** Displays binarized accuracy scores (1-2 → Incorrect, 3-4 → Correct) by:
- Teacher QA (human-written questions)
- Synthetic QA (AI-generated questions)
- Individual QA sources
- Per-judge breakdown

## File Structure

```
DrawEduMath/
├── data/
│   └── AllImages/Resized_Merged_Problem_Images/  # Student work images (from HuggingFace)
├── output/                                        # Model CSV files
│   ├── template.csv                               # Template CSV
│   ├── {model_name}.csv                           # Model results
│   └── {judge}_judge/                             # Judge batch outputs
├── logs/                                          # Execution logs
│   └── {model_name}/
│       ├── generation.log
│       ├── judge_claude.log
│       ├── judge_gemini.log
│       └── judge_gpt4o.log
└── scripts/pipeline/
    ├── generation/                                # Generation scripts
    │   ├── generate_anthropic.py
    │   ├── generate_google.py
    │   ├── generate_openai.py
    │   └── generate_together.py
    ├── judges/                                    # Judge scripts
    │   ├── judge_claude.py
    │   ├── judge_gemini.py
    │   ├── judge_gpt4o.py
    │   └── merge_judge.py
    ├── evaluation/                                # Evaluation scripts
    │   ├── run_evaluation.py
    │   └── print_scores.py
    ├── prompts.py                                 # Prompt templates
    ├── shared_utils.py                            # Shared utilities
    └── requirements.txt                           # Python dependencies
```

## CSV Format

### Input Columns

- `QA_Pair_ID` - Unique identifier
- `Model Name` - Model identifier
- `Image Name` - Filename in AllImages folder
- `Question` - Question text
- `Reference Answer` - Ground truth answer
- `QA Type` - Question type (teacher/gpt4o/claude)

### Generated Columns

- `Model Answer` - Model's response
- `Claude_Judge_Rating` - Rating 1-4 (or -1)
- `Claude_Judge_Reason` - Explanation
- `Gemini_Judge_Rating` - Rating 1-4 (or -1)
- `Gemini_Judge_Reason` - Explanation
- `Openai_Judge_Rating` - Rating 1-4 (or -1)
- `Openai_Judge_Reason` - Explanation
- `Ensemble_Judge_Rating` - Mode of three judges
- `BERTScore F1` - (optional) BERTScore F1 metric
- `ROUGEL` - (optional) ROUGE-L metric

## Adding New Models

1. Add to `AVAILABLE_MODELS` dict in generation script:

```python
"your-model-key": {
    "api_name": "actual-api-model-name",
    "display_name": "Display Name",
    "csv_name": "output_filename"
}
```

2. Set `SELECTED_MODEL = "your-model-key"`
3. Run generation script
4. Follow judging workflow (steps 2-4)

## Dataset Details

- **56,054 QA pairs** evaluating VLM understanding of student math work
- **QA Types**:
  - `teacher` - Human-written questions by educators
  - `gpt4o` / `claude` - Synthetically generated questions
- **Evaluation**: Binarized accuracy (1-2 → Incorrect, 3-4 → Correct)
- **Scoring**: Reported separately for teacher QA vs synthetic QA

## Judge Methodology

### Triple-Judge Ensemble

Each model response is independently evaluated by three judge models:
1. **Claude 4.5 Sonnet** (claude-sonnet-4-5)
2. **Gemini 2.5 Pro** (gemini-2.5-pro)
3. **GPT-4o** (gpt-4o)

The final `Ensemble_Judge_Rating` is the **mode** (most common rating) across all three judges.

### Rating Guidelines

Judges evaluate whether the model answer matches the reference answer:
- **4 (Perfect):** Semantically identical, same conclusion
- **3 (Valid Alternative):** Different approach but mathematically correct
- **2 (Major Error):** Factually wrong or missing critical information
- **1 (Irrelevant):** Off-topic or completely incorrect

### Batch Processing

- Batch size: 1,000 QA pairs per batch
- Uses provider batch APIs for efficient processing
- Automatic checkpointing and resume support
- Skips already-judged QA pairs

## Prompt Templates

### Generation Prompt

Located in `scripts/pipeline/prompts.py` as `GENERATE_ANSWER_PROMPT`.

Instructs VLM to:
- Analyze handwritten work on right side of concatenated image
- Answer question based on visible student work
- Provide clear, concise responses

### Judge Prompt

Located in `scripts/pipeline/prompts.py` as `JUDGE_PROMPT_TEMPLATE`.

Provides:
- Question text
- Reference answer (ground truth)
- Model answer to evaluate
- Rating scale (1-4)
- JSON output format: `{"rating": X, "reason": "..."}`

## Troubleshooting

### Rate Limits

If you hit API rate limits:
- Adjust `RATE_LIMIT` in generation scripts
- Increase `SLEEP_TIME` for longer pauses
- Scripts auto-checkpoint and can be resumed

### Missing Images

Ensure images are downloaded from HuggingFace:
```bash
# Dataset URL
https://huggingface.co/datasets/allenai/DrawEduMath

# Expected location
data/AllImages/Resized_Merged_Problem_Images/
```

### CSV Errors

If CSV files are corrupted:
- Check logs in `logs/{model_name}/` for error messages
- Use `output/template.csv` as clean starting point
- Ensure all column names match expected format

### Import Errors

If Python imports fail:
```bash
cd scripts/pipeline
python -c "import sys; sys.path.insert(0, '.'); import prompts; import shared_utils"
```

All imports are relative from the pipeline directory.

## Performance Tips

1. **Parallel Judging:** Run all three judge scripts simultaneously for faster evaluation
2. **Checkpointing:** Generation and judging auto-save progress every 10 requests
3. **Incremental Evaluation:** Scripts skip already-processed rows
4. **Batch API:** Use batch APIs (Claude, OpenAI, Gemini) for cost-effective large-scale judging

## Citation

If you use this pipeline, please cite:

```
@inproceedings{baral2024drawedumath,
  author    = {Baral, Sami and Li, Lucy and Knight, Ryan and Ng, Alice and Soldainin, Luca and Heffernan, Neil and Lo, Kyle},
  title     = {DrawEduMath: Evaluating Vision Language Models with Expert-Annotated Students' Hand-Drawn Math Images},
  booktitle = {The 4th Workshop on Mathematical Reasoning and AI at NeurIPS'24},
  year      = {2024}
}
```

## Links

- **Dataset:** https://huggingface.co/datasets/allenai/DrawEduMath
- **Website:** https://drawedumath.org
- **Repository:** https://github.com/allenai/drawedumath
- **Paper:** Coming soon
