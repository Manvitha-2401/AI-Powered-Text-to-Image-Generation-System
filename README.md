# 🌸 AI Powered Text-to-Image Generation System

An AI-powered Text-to-Image Generation System developed as part of my AI internship project. This project demonstrates the complete workflow of generating images from textual descriptions using Deep Learning, Generative Adversarial Networks (GANs), Hugging Face Transformers, and Attention Mechanisms.

---

## 📌 Project Overview

The project simulates a real-world text-to-image generation pipeline by integrating multiple AI components, including:

- Fine-Tuning a Stable Diffusion model
- Conditional Generative Adversarial Network (CGAN)
- Text preprocessing using Hugging Face Transformers (CLIP)
- Dataset analysis and visualization
- Cross-Attention mechanism
- Complete end-to-end text-to-image generation pipeline

---

## 🎯 Objectives

- Fine-tune a pre-trained text-to-image model.
- Generate images conditioned on textual descriptions.
- Learn conditional image generation using CGAN.
- Generate text embeddings using CLIP.
- Improve image generation quality using Cross-Attention.
- Build a complete text-to-image generation workflow.

---

# 📂 Project Structure

```text
TEXT-TO-IMAGE-GENERATOR
│
├── app.py
├── data/
├── images_task1/
├── models/
├── notebooks/
├── outputs/
├── report/
├── src/
├── README.md
├── requirements.txt
└── .gitignore
```

---

# 🛠 Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- Diffusers
- Stable Diffusion
- LoRA
- Conditional GAN (CGAN)
- Cross Attention
- NumPy
- Pandas
- Matplotlib
- Pillow
- Google Colab
- VS Code

---

# 📖 Task-wise Implementation

## ✅ Task 1 – Fine-Tuning Stable Diffusion

- Fine-tuned a pre-trained Stable Diffusion model.
- Applied LoRA (Low-Rank Adaptation).
- Generated flower images after fine-tuning.

---

## ✅ Task 2 – Conditional GAN (CGAN)

- Built a Conditional GAN from scratch.
- Generated Circle and Square images using class labels.
- Implemented Generator and Discriminator networks.
- Saved trained models.

---

## ✅ Task 3 – Text Preprocessing

- Used Hugging Face CLIP Tokenizer.
- Converted text descriptions into tokenized sequences.
- Generated text embeddings for image generation.

---

## ✅ Task 4 – Dataset Analysis

Performed Exploratory Data Analysis (EDA):

- Image size analysis
- Width distribution
- Height distribution
- Class distribution
- Sample image visualization

---

## ✅ Task 5 – Attention Strategy

Implemented Cross-Attention using Multi-Head Attention.

Features:

- Cross-Attention Layer
- Text-Image Feature Alignment
- Improved Generator Performance

---

## ✅ Task 6 – Complete Text-to-Image Pipeline

Integrated all previous tasks into one complete workflow.

Pipeline:

Text Prompt

↓

CLIP Tokenizer

↓

Text Embeddings

↓

Cross Attention

↓

Attention-Based Generator

↓

Generated Image

---

# 📊 Outputs

The project generates:

- Fine-tuned flower images
- CGAN generated shapes
- Attention-based flower images
- Dataset analysis graphs

---

# 📁 Models

The project contains trained models:

- Generator
- Discriminator
- Attention Generator
- LoRA Weights

---

# 🚀 Installation

Clone the repository

```bash
git clone <repository-url>
```

Move into the project folder

```bash
cd TEXT-TO-IMAGE-GENERATOR
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

Open the notebooks inside the **notebooks/** folder to execute each task individually.

Python scripts are available inside the **src/** folder for modular implementation.

---

# 📈 Results

Successfully implemented:

- Stable Diffusion Fine-Tuning
- Conditional GAN
- Text Embeddings using CLIP
- Cross-Attention
- Complete Text-to-Image Generation Pipeline

---

# 🔮 Future Improvements

- Train on larger datasets
- Improve image resolution
- Deploy as a web application
- Integrate real-time prompt generation
- Use advanced diffusion models

---

# 👩‍💻 Author

**Donthi Reddy Manvitha Reddy**

B.Tech – Computer Science and Engineering (AI & ML)

AI Internship Project

---

# 📜 License

This project was developed for educational and internship purposes.