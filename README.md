# DHC-Phase-2-ML-Task_1
Fine-tuning bert-base-uncased on the AG News dataset to classify news headlines into 4 topic categories, with a live Gradio web interface for real-time inference.
# News Topic Classifier — BERT Fine-Tuning

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.10-red) ![Transformers](https://img.shields.io/badge/HuggingFace-Transformers_5.0-yellow) ![Gradio](https://img.shields.io/badge/Gradio-4.0+-orange) ![License](https://img.shields.io/badge/License-MIT-green)

---

## Objective

Fine-tune a pretrained transformer model (BERT) to classify news headlines into one of four topic categories using the AG News benchmark dataset. The project covers the full pipeline — from raw data exploration through tokenization, model training, evaluation, and deployment as a live web app.

---

## Methodology / Approach

The project is structured into five sequential phases, each building on the last.

### Phase 1 — Data Exploration
Before writing any model code, the dataset was explored to understand its structure. AG News contains 120,000 training and 7,600 test samples across 4 perfectly balanced classes (30,000 each), meaning no class-weighting tricks were needed. Text length analysis on 5,000 samples showed an average of 39.2 words and a max of 134 — confirming that `max_length=128` tokens covers 99.9% of samples without truncation loss.

### Phase 2 — Tokenization
The `bert-base-uncased` WordPiece tokenizer was applied to all samples. Each text was padded or truncated to exactly 128 tokens, producing three tensors per sample: `input_ids`, `attention_mask`, and `token_type_ids`. The tokenized dataset was saved to Google Drive to avoid re-tokenizing across Colab sessions.

### Phase 3 — Fine-Tuning
A pretrained `bert-base-uncased` model was loaded with a randomly initialized 4-class classification head (`Linear(768, 4)`). The `[CLS]` token embedding — which BERT uses to encode sentence-level meaning — feeds into this head to produce the class logits. Training ran for 3 epochs using the Hugging Face `Trainer` API with the following configuration:

| Hyperparameter | Value |
|---|---|
| Learning rate | 2e-5 |
| Batch size | 32 |
| Epochs | 3 |
| Warmup steps | 1,125 (10% of total) |
| Weight decay | 0.01 |
| Precision | fp16 (mixed) |

Checkpoints were saved to Google Drive after every epoch, enabling automatic resume if the Colab GPU session disconnected mid-training.

### Phase 4 — Evaluation
The trained model was evaluated on the 7,600-sample test set using accuracy, macro F1, a per-class classification report, and a confusion matrix.

### Phase 5 — Deployment
A live Gradio web interface was built around the trained model, allowing real-time headline classification with confidence scores for all four classes.

---

## Results

### Overall Metrics

| Metric | Value |
|--------|-------|
| Accuracy | **95.01%** |
| F1 Score (macro) | **0.9502** |
| Training loss | 0.1705 |
| Training time | ~37 min on T4 GPU |

### Per-Class Performance

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| World | 0.97 | 0.96 | 0.96 |
| Sports | 0.99 | 0.99 | **0.99** |
| Business | 0.93 | 0.91 | 0.92 |
| Sci/Tech | 0.91 | 0.94 | 0.93 |

### Confusion Matrix

```
              World    Sports   Business  Sci/Tech
World         1,816        9        40        35
Sports           12    1,877         5         6
Business         30        7     1,734       129
Sci/Tech         18        8       109     1,765
```

### Key Observations

- **Sports is the easiest class** (F1 = 0.99) — sports headlines have distinctive vocabulary with almost no overlap with other categories.
- **Business and Sci/Tech are most confused** — 129 Business articles were predicted as Sci/Tech and 109 Sci/Tech as Business. This makes intuitive sense: technology companies appear in both categories (e.g. "Apple reports earnings" vs "Apple launches new chip").
- **World has minor bleed into Business** — 40 World articles were misclassified as Business, likely due to headlines covering geopolitical economic events.

---

## Project Structure

```
news_classifier/
│
├── phase1_explore_data.py   # AG News exploration — class balance, text lengths
├── phase2_tokenize.py       # WordPiece tokenization, padding, saving to disk
├── phase3_train.py          # BERT fine-tuning with checkpoint resume support
├── phase4_evaluate.py       # Accuracy, F1, confusion matrix, manual tests
├── phase5_deploy.py         # Live Gradio web app
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/news-topic-classifier.git
cd news-topic-classifier
pip install -r requirements.txt
```

### 2. Run phases in order

```bash
python phase1_explore_data.py   # Explore data
python phase2_tokenize.py       # Tokenize and save (~30 sec)
python phase3_train.py          # Fine-tune BERT (~20 min on GPU)
python phase4_evaluate.py       # Evaluate
python phase5_deploy.py         # Launch Gradio app
```

---

## Running on Google Colab

This project was developed on Google Colab with a free T4 GPU.

1. Go to **Runtime → Change runtime type → T4 GPU**
2. Mount Google Drive at the start of each session:

```python
from google.colab import drive
drive.mount("/content/drive")
```

3. All checkpoints save to Drive automatically. If your session disconnects mid-training, re-running the script resumes from the last saved epoch — no work is lost.

---

## Skills Demonstrated

| Skill | Where |
|-------|-------|
| Transfer learning with BERT | Phase 3 |
| WordPiece tokenization & attention masks | Phase 2 |
| Hugging Face Trainer API | Phase 3 |
| Evaluation: accuracy, F1, confusion matrix | Phase 4 |
| Checkpoint saving & resume logic | Phase 3 |
| Gradio deployment | Phase 5 |

---

## Tech Stack

[Hugging Face Transformers](https://huggingface.co/docs/transformers) · [Hugging Face Datasets](https://huggingface.co/docs/datasets) · [PyTorch](https://pytorch.org/) · [scikit-learn](https://scikit-learn.org/) · [Gradio](https://gradio.app/)

---

## License

MIT
