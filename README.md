# 🧠 Hybrid Mental Health Detection System

A **hybrid NLP-based text classification system** designed to analyze textual input and identify patterns associated with **depression, anxiety, and happiness**.

The project combines traditional machine learning with modern Transformer-based sentence embeddings to compare different approaches to text classification.

> ⚠️ **Important:** This project is an educational/research prototype and is **not a medical diagnostic tool**. Predictions should not be treated as professional mental-health diagnoses or medical advice.

---

## 🚀 Project Overview

Mental-health-related language can contain subtle patterns that are difficult to identify using simple keyword-based systems.

This project explores how Natural Language Processing can be used to analyze text and classify it into predefined categories.

The system implements two approaches:

1. **TF-IDF + Logistic Regression**
2. **Sentence-BERT (SBERT) embeddings + classification**

The results are presented through an interactive **Gradio interface**.

---

## 🎯 Objectives

The main objectives of this project are:

* Analyze user-provided text using NLP techniques.
* Extract meaningful linguistic features from text.
* Compare traditional ML with Transformer-based representations.
* Classify text into predefined emotional/mental-health-related categories.
* Build an easy-to-use interactive interface.
* Explore how semantic embeddings can improve text representation.

---

## 🧠 Classification Categories

The system works with three project-defined categories:

| Category   | Description                                                     |
| ---------- | --------------------------------------------------------------- |
| Depression | Text patterns associated with depressive language               |
| Anxiety    | Text patterns associated with anxious or worry-related language |
| Happiness  | Text patterns associated with positive/happy language           |

These categories represent **classification labels in the dataset**, not clinical diagnoses.

---

# 🏗️ System Architecture

```text
                    User Input
                        │
                        ▼
              ┌───────────────────┐
              │  Gradio Interface │
              └─────────┬─────────┘
                        │
                        ▼
                Text Preprocessing
                        │
                        ▼
              ┌───────────────────┐
              │   NLP Pipeline    │
              └─────────┬─────────┘
                        │
              ┌─────────┴─────────┐
              │                   │
              ▼                   ▼
       TF-IDF Pipeline      SBERT Pipeline
              │                   │
              ▼                   ▼
     Logistic Regression    Sentence Embeddings
              │                   │
              └─────────┬─────────┘
                        ▼
                 Classification
                        │
                        ▼
               Predicted Category
                        │
                        ▼
                 Gradio Output
```

---

# 🔬 Methodology

## 1. Text Preprocessing

Before training the models, the text is processed to create cleaner input.

Typical preprocessing steps include:

* Text normalization
* Lowercasing
* Removing unnecessary characters
* Tokenization
* Handling unwanted text patterns
* Preparing text for feature extraction

The exact preprocessing pipeline can depend on the dataset and implementation.

---

# 2. TF-IDF + Logistic Regression

The first approach uses a traditional NLP pipeline.

### TF-IDF

**TF-IDF (Term Frequency–Inverse Document Frequency)** converts text into numerical features.

It assigns importance to words based on:

* How frequently a word occurs in a document
* How frequently that word occurs across the dataset

The basic idea is:

```text
Text
 ↓
Tokenization
 ↓
TF-IDF Vectorization
 ↓
Numerical Feature Matrix
 ↓
Logistic Regression
 ↓
Prediction
```

### Logistic Regression

Logistic Regression is used as the classification algorithm.

It learns relationships between TF-IDF features and the predefined classes.

---

# 3. Sentence-BERT

The second approach uses **Sentence-BERT (SBERT)**.

Unlike basic word-frequency features, SBERT generates dense vector representations that capture more semantic information from sentences.

```text
Input Text
    ↓
Sentence-BERT
    ↓
Dense Embedding
    ↓
Feature Representation
    ↓
Classifier
    ↓
Prediction
```

This allows the system to represent semantically similar sentences more effectively than simple keyword-frequency representations.

---

# 🔀 Hybrid Approach

The project is called a **Hybrid Mental Health Detection System** because it explores both traditional and Transformer-based NLP approaches.

### Traditional NLP

```text
TF-IDF
   ↓
Logistic Regression
   ↓
Prediction
```

### Transformer NLP

```text
SBERT
   ↓
Semantic Embedding
   ↓
Classifier
   ↓
Prediction
```

The two approaches provide different representations of language:

| Approach            | Strength                                           |
| ------------------- | -------------------------------------------------- |
| TF-IDF              | Simple, fast, interpretable                        |
| Logistic Regression | Strong baseline classifier                         |
| SBERT               | Captures semantic meaning                          |
| Hybrid comparison   | Allows evaluation of different NLP representations |

---

# 🛠️ Technology Stack

## Programming Language

* Python

## Machine Learning

* Scikit-learn
* Logistic Regression
* TF-IDF Vectorization

## Deep Learning / NLP

* Sentence-BERT
* Transformer-based embeddings

## Data Processing

* Pandas
* NumPy

## Interface

* Gradio

## Development

* Jupyter Notebook
* Python
* Git
* GitHub

---

# 📁 Project Structure

```text
Hybrid-Mental-Health-Detection/
│
├── data/
│   └── dataset.csv
│
├── models/
│   └── trained_models/
│
├── notebooks/
│   └── mental_health_detection.ipynb
│
├── app.py
├── requirements.txt
├── .gitignore
├── README.md
└── LICENSE
```

> Adjust the folder names above if your actual repository structure is different.

---

# 📊 Machine Learning Pipeline

```text
Dataset
   │
   ▼
Data Cleaning
   │
   ▼
Text Preprocessing
   │
   ├─────────────────────┐
   ▼                     ▼
TF-IDF               SBERT
   │                     │
   ▼                     ▼
Feature Vectors      Embeddings
   │                     │
   ▼                     ▼
Logistic Regression  Classification
   │                     │
   └──────────┬──────────┘
              ▼
       Model Evaluation
              │
              ▼
       Gradio Application
```

---

# 📈 Model Evaluation

The models can be evaluated using standard classification metrics:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

### Why F1-score?

For text classification, accuracy alone may not provide the complete picture, especially when classes are imbalanced.

The F1-score combines:

```text
Precision + Recall
        ↓
     F1 Score
```

This provides a useful overall measure of classification performance.

---

# 💻 Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git

cd Hybrid-Mental-Health-Detection
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Application

Run the Gradio application:

```bash
python app.py
```

The application will provide a local Gradio interface where users can enter text and receive a predicted category.

---

# 🖥️ Example

### Input

```text
I have been feeling very worried about everything
and I cannot stop thinking about what might go wrong.
```

### Example Output

```text
Predicted Category: Anxiety
```

The output represents the model's classification based on patterns learned from the training dataset.

It should **not** be interpreted as a clinical assessment.

---

# 🌟 Key Features

### 🔹 NLP-Based Classification

Uses natural-language processing to analyze textual input.

### 🔹 Traditional ML Baseline

TF-IDF with Logistic Regression provides a strong and lightweight baseline.

### 🔹 Transformer-Based Representation

Sentence-BERT provides semantic sentence embeddings.

### 🔹 Interactive Interface

Gradio allows users to interact with the model without requiring a command-line workflow.

### 🔹 Model Comparison

The project demonstrates the difference between traditional feature engineering and Transformer-based representations.

---

# 🔐 Privacy & Responsible AI

Because the application can process potentially sensitive text, privacy should be considered when deploying it.

Recommended practices include:

* Do not store user text unnecessarily.
* Do not expose personal information in logs.
* Avoid collecting identifiable user information.
* Clearly communicate that the model is experimental.
* Do not use predictions as a substitute for professional assessment.

---

# ⚠️ Disclaimer

This project is intended **only for educational and research purposes**.

The system predicts predefined text categories based on patterns learned from its dataset. It does not have the ability to clinically diagnose depression, anxiety, or any other mental-health condition.

If someone is experiencing mental-health difficulties, they should seek appropriate support from a qualified healthcare professional rather than relying on an AI prediction.

---

# 🔮 Future Improvements

Potential improvements include:

* Fine-tuning Transformer models
* Testing additional Transformer architectures
* Better handling of class imbalance
* Explainable AI for predictions
* Confidence calibration
* Cross-validation
* More diverse datasets
* Multilingual text support
* Bias and fairness evaluation
* Model monitoring
* Privacy-preserving inference
* Human-in-the-loop review
* More robust evaluation on unseen datasets

---

# 📚 Learning Outcomes

This project demonstrates practical experience with:

* Natural Language Processing
* Text preprocessing
* Feature engineering
* TF-IDF
* Logistic Regression
* Transformer models
* Sentence-BERT
* Semantic embeddings
* Classification
* Model evaluation
* Gradio application development
* Responsible AI considerations

---

# 👨‍💻 Author

**Harsh Modi**

MSc Data Science

### Areas of Interest

* Machine Learning
* Deep Learning
* Natural Language Processing
* Generative AI
* Retrieval-Augmented Generation
* AI Applications

---

## ⭐ Project Highlights

```text
Traditional NLP
      +
Transformer NLP
      +
Machine Learning
      +
Interactive Gradio UI
      =
Hybrid Mental Health Detection System
```

If you find the project useful, consider giving the repository a ⭐ on GitHub.
