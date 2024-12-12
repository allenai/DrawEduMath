"""

Bar plot of VLM performances across question types.

@Kylel

"""

from textwrap import wrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Create DataFrame from the data
metrics = [
    "Correctness & errors",
    "Counting content",
    "Higher-level understanding",
    "Image creation & medium",
    "Low-level characteristics",
    "Problem strategy & solution",
    "Writing & labels",
]

# Wrap metric names
wrapped_metrics = ["\n".join(wrap(m, width=15)) for m in metrics]

gpt4_robot = [0.525, 0.642, 0.696, 0.886, 0.674, 0.758, 0.711]
gpt4_human = [0.559, 0.671, 0.599, float("nan"), 0.624, 0.719, 0.606]
claude_robot = [0.491, 0.516, 0.642, 0.805, 0.635, 0.660, 0.647]
claude_human = [0.610, 0.667, 0.605, float("nan"), 0.660, 0.740, 0.620]
gemini_robot = [0.601, 0.602, 0.632, 0.795, 0.566, 0.716, 0.615]
gemini_human = [0.440, 0.578, 0.484, float("nan"), 0.457, 0.539, 0.499]
llama_robot = [0.402, 0.247, 0.333, 0.589, 0.402, 0.406, 0.338]
llama_human = [0.276, 0.265, 0.350, float("nan"), 0.369, 0.307, 0.216]

# Set up the plot
plt.figure(figsize=(15, 10))
bar_width = 0.1
index = np.arange(len(metrics))

# Create bars with lighter robot bars and darker human bars
plt.bar(
    index - 3 * bar_width,
    gpt4_robot,
    bar_width,
    label="GPT-4o (Synthetic)",
    color="#1f77b4",
    alpha=0.5,
)
plt.bar(
    index - 2 * bar_width,
    gpt4_human,
    bar_width,
    label="GPT-4o (Human)",
    color="#1f77b4",
)
plt.bar(
    index - bar_width,
    claude_robot,
    bar_width,
    label="Claude 3.5 (Synthetic)",
    color="#2ca02c",
    alpha=0.5,
)
plt.bar(index, claude_human, bar_width, label="Claude 3.5 (Human)", color="#2ca02c")
plt.bar(
    index + bar_width,
    gemini_robot,
    bar_width,
    label="Gemini 1.5 (Synthetic)",
    color="#ff7f0e",
    alpha=0.5,
)
plt.bar(
    index + 2 * bar_width,
    gemini_human,
    bar_width,
    label="Gemini 1.5 (Human)",
    color="#ff7f0e",
)
plt.bar(
    index + 3 * bar_width,
    llama_robot,
    bar_width,
    label="Llama 3.2 (Synthetic)",
    color="#d62728",
    alpha=0.5,
)
plt.bar(
    index + 4 * bar_width,
    llama_human,
    bar_width,
    label="Llama 3.2 (Human)",
    color="#d62728",
)

# Customize the plot
plt.xlabel("Type of Question", fontsize=18)
plt.ylabel("Accuracy", fontsize=18)

# Increase font size for tick labels
plt.xticks(index + bar_width / 2, wrapped_metrics, fontsize=14)
plt.yticks(fontsize=18)

# Move legend to top, horizontal orientation
plt.legend(bbox_to_anchor=(0.5, 1.15), loc="center", ncol=4, fontsize=11)

plt.grid(True, axis="y", linestyle="--", alpha=0.7)

plt.legend(bbox_to_anchor=(0.5, 1.08), loc="center", ncol=4, fontsize=14)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save with transparent background
plt.savefig("vlm_performance_bar.png", dpi=300, bbox_inches="tight", transparent=True)
