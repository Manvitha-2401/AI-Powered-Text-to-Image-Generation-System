"""
--------------------------------------------------------
Task 4: Dataset Analysis
AI-Powered Text-to-Image Generation System

Author: Donthi Reddy Manvitha Reddy
Internship Project

Objective:
Analyze the flower image dataset by calculating image
dimensions, class distribution, and displaying sample
images.
--------------------------------------------------------
"""

import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def load_images(image_folder):
    """Load all image filenames."""

    images = sorted(os.listdir(image_folder))

    print(f"Total Images: {len(images)}")

    return images


def image_statistics(image_folder, images):
    """Calculate width and height statistics."""

    widths = []
    heights = []

    for img_name in images:

        img = Image.open(os.path.join(image_folder, img_name))

        w, h = img.size

        widths.append(w)
        heights.append(h)

    print("\nImage Statistics")
    print("---------------------------")
    print("Minimum Width :", min(widths))
    print("Maximum Width :", max(widths))
    print("Average Width :", np.mean(widths))

    print()

    print("Minimum Height:", min(heights))
    print("Maximum Height:", max(heights))
    print("Average Height:", np.mean(heights))

    return widths, heights


def plot_width_distribution(widths):

    plt.figure(figsize=(6,4))

    plt.hist(widths, bins=20)

    plt.title("Width Distribution")

    plt.xlabel("Width")

    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.show()


def plot_height_distribution(heights):

    plt.figure(figsize=(6,4))

    plt.hist(heights, bins=20)

    plt.title("Height Distribution")

    plt.xlabel("Height")

    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":

    IMAGE_FOLDER = "data/images"

    images = load_images(IMAGE_FOLDER)

    widths, heights = image_statistics(
        IMAGE_FOLDER,
        images
    )

    plot_width_distribution(widths)

    plot_height_distribution(heights)
    