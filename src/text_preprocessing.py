"""
--------------------------------------------------------
Task 3: Text Preprocessing using Hugging Face Transformers
AI-Powered Text-to-Image Generation System

Author: Donthi Reddy Manvitha Reddy
Internship Project

Objective:
Convert textual descriptions into tokenized and encoded
representations using Hugging Face Transformers.
--------------------------------------------------------
"""

from transformers import CLIPTokenizer


def load_tokenizer():
    """Load the pretrained DistilBERT tokenizer."""

    print("=" * 60)
    print("Loading Hugging Face Tokenizer...")
    print("=" * 60)

    tokenizer = CLIPTokenizer.from_pretrained(
    "openai/clip-vit-base-patch32"
)
    

    print("✅ Tokenizer loaded successfully!\n")

    return tokenizer


def create_text_descriptions():
    """Sample flower descriptions."""

    return [
        "A beautiful red flower.",
        "A bright yellow sunflower.",
        "A white daisy in the garden.",
        "A colorful tulip blooming in spring.",
        "A purple orchid with delicate petals."
    ]


def tokenize_text(tokenizer, descriptions):
    """Tokenize text descriptions."""

    encoded = tokenizer(
        descriptions,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    return encoded


if __name__ == "__main__":

    tokenizer = load_tokenizer()

    descriptions = create_text_descriptions()

    print("Sample Text Descriptions\n")

    for i, text in enumerate(descriptions, start=1):
        print(f"{i}. {text}")

    encoded = tokenize_text(tokenizer, descriptions)

    print("\n" + "=" * 60)
    print("TOKENIZED OUTPUT")
    print("=" * 60)

    print("\nInput IDs:\n")
    print(encoded["input_ids"])

    print("\nAttention Mask:\n")
    print(encoded["attention_mask"])