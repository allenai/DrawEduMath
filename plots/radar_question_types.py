"""

Make radar plot of question types.

@kylel

"""

import matplotlib.pyplot as plt
import numpy as np


def create_comparison_radar_plot(categories, synthetic_values, teacher_values):
    """
    Create a radar/spider plot comparing two sets of values with custom styling.
    """
    # Number of variables
    num_vars = len(categories)

    # Compute angle for each axis
    angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
    angles += angles[:1]  # Complete the circle

    # Initialize the spider plot
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection="polar"))

    # Plot data
    synthetic_values_plot = np.concatenate((synthetic_values, [synthetic_values[0]]))
    teacher_values_plot = np.concatenate((teacher_values, [teacher_values[0]]))

    # Custom colors
    synthetic_color = "#6FE0BA"  # A medium green
    teacher_color = "#F0529C"  # A medium pink

    # Plot both lines
    ax.plot(
        angles,
        synthetic_values_plot,
        "o-",
        linewidth=2,
        label="Synthetic (AI)",
        color=synthetic_color,
    )
    ax.fill(angles, synthetic_values_plot, alpha=0.25, color=synthetic_color)

    ax.plot(
        angles,
        teacher_values_plot,
        "o-",
        linewidth=2,
        label="Teacher",
        color=teacher_color,
    )
    ax.fill(angles, teacher_values_plot, alpha=0.25, color=teacher_color)

    # Draw axis lines for each angle
    ax.set_xticks(angles[:-1])

    # Remove default labels
    ax.set_xticklabels([])

    # Add custom labels outside the plot
    for idx, (angle, label) in enumerate(zip(angles[:-1], categories)):
        # Convert angle to degrees for text rotation
        angle_deg = np.degrees(angle)

        # Adjust label alignment and position based on angle
        if angle_deg <= 90:
            ha = "left"
            va = "bottom"
        elif angle_deg <= 180:
            ha = "right"
            va = "bottom"
        elif angle_deg <= 270:
            ha = "right"
            va = "top"
        else:
            ha = "left"
            va = "top"

        # Position text at 33 (adjust this value to move labels further out or closer)
        ax.text(angle, 33, label, ha=ha, va=va, size=14)

    # Add gridlines with custom size
    ax.set_rlabel_position(0)
    plt.yticks(
        [5, 10, 15, 20, 25, 30],
        ["5", "10", "15", "20", "25", "30"],
        color="grey",
        size=14,
    )
    plt.ylim(0, 32)  # Increased ylim to accommodate labels

    # Add legend with custom size
    plt.legend(bbox_to_anchor=(0.5, 1.08), loc="center", ncol=4, fontsize=18)

    # Make the plot more compact
    plt.tight_layout()

    return fig, ax


# Data
categories = [
    "Writing and\nlabels",
    "Higher-level\nunderstanding of\nmath",
    "Correctness and\nerrors",
    "Image creation\nand medium",
    "Counting content",
    "Problem solving\nsteps, strategy,\nand solution",
    "Low-level\ncomposition and\npositioning",
]

# Average of Claude and GPT-4
synthetic_values = [
    (14.6 + 16.1) / 2,  # Writing
    (26.7 + 25.7) / 2,  # Higher-level
    (1.7 + 1.5) / 2,  # Correctness
    (15.0 + 16.0) / 2,  # Image creation
    (10.5 + 9.1) / 2,  # Counting
    (9.2 + 10.5) / 2,  # Problem solving
    (21.9 + 20.0) / 2,  # Low-level
]

teacher_values = [17.3, 18.8, 23.0, 0.0, 5.7, 23.2, 11.4]

# Create the plot
fig, ax = create_comparison_radar_plot(
    categories=categories,
    synthetic_values=synthetic_values,
    teacher_values=teacher_values,
)

# Save with transparent background
plt.savefig("question_types_radar.png", dpi=300, bbox_inches="tight", transparent=True)
