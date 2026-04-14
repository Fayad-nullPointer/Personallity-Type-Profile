# Personality Type Profile Prediction

A deep learning and natural language processing (NLP) project aimed at predicting a user's Myers-Briggs Type Indicator (MBTI) personality profile based on their textual posts.

## 📖 Project Overview

This project develops an advanced hierarchical Deep Learning model using PyTorch to analyze a user's text and classify their personality across the 4 MBTI dimensions:
- **E/I** - Extroversion vs. Introversion
- **N/S** - Intuition vs. Sensing
- **T/F** - Thinking vs. Feeling
- **J/P** - Judging vs. Perceiving

### 🧠 Core Innovation: Cluster-Based Embedding with Attention

The standout feature of this repository is the custom **Cluster Attention Mechanism**. Unlike standard models that simply average text embeddings, this model uses a bidirectional LSTM at the word level to encode individual user posts. These posts are then aggregated to create a dense user profile vector, which attends over **16 learnable prototype vectors** (representing the 16 unique MBTI types). This allows the model to learn the fundamental structure of each MBTI personality and softly align unseen users to the most similar types.

## ✨ Features and Methodology

* **Advanced Text Preprocessing**: Customized cleaning pipelines using `spacy`, `nltk`, and `contractions` to handle punctuation, links, and text normalization while preserving critical structural tokens (e.g., post separators `|||`).
* **Word Embeddings**: Utilizes pre-trained **GloVe** (100d) word vectors to capture rich semantic meaning from user text.
* **Hierarchical Neural Network Architecture**:
  * **Level 1 (Word-LSTM)**: A Bidirectional LSTM processes the word embeddings for each individual post.
  * **Level 2 (Aggregation)**: Aggregates multiple posts from a single user into a unified person vector using Attention pooling.
  * **Level 2.5 (Cluster Attention)**: Queries the unified person against 16 MBTI prototype vectors to build a cluster context.
  * **Level 3 (Classification)**: A Feed-Forward Network that outputs 4 distinct logits representing the 4 independent MBTI axes.
* **Imbalanced Data Handling**: Automatically computes and applies class weights for the BCE loss to tackle the natural class imbalances present in MBTI datasets.
* **Threshold Tuning**: Finds the optimal decision thresholds per MBTI dimension based on the F1-score to maximize predictive performance.

## 📁 Repository Structure

- `cluster-approach.ipynb` / `main.ipynb` - Jupyter notebooks containing the end-to-end data processing, custom PyTorch model construction (Cluster Attention LSTM), training, and evaluation logic.
- `Basic.py` / `Transformer.py` / `script.py` - Core scripts housing architectural variants, helper functions, and experimental transformer models.
- `Evalaute.py` - Evaluation scripts to test model performance and output threshold-tuned predictions.
- `data/` - Contains the primary dataset (`Lab 3 - Personality Profile Type.csv`).
- `Processed_train.csv` / `Preprocessed_test.csv` - Cleaned and formatted inputs ready for robust training and inference.

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.8+ installed along with the required libraries:

```bash
pip install pandas numpy torch scikit-learn spacy nltk contractions tqdm matplotlib seaborn requests bs4
python -m spacy download en_core_web_sm
```

### Running the Model

1. Ensure your dataset is located in the `data/` directory.
2. You can explore the training pipeline directly using the provided Jupyter notebooks:
   ```bash
   jupyter notebook cluster-approach.ipynb
   ```
3. The notebook will automatically download the necessary **GloVe** embeddings, preprocess the text, train the PyTorch model, dynamically tune inference thresholds, and yield the final test accuracy across all 4 MBTI dimensions.
