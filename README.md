# Sentiment Analysis on Product Reviews

A complete data science project for analyzing customer sentiment from Flipkart product reviews, featuring a machine learning pipeline and an interactive Streamlit web application.

## 🚀 Live Demo

Check out the live application here: **[https://sentiment-projec.streamlit.app/](https://sentiment-projec.streamlit.app/)**

---

## Project Status

✅ **Step 1: Data Collection** - COMPLETE
✅ **Step 2: Data Cleaning & Preprocessing** - COMPLETE
✅ **Step 3: Feature Extraction** - COMPLETE
✅ **Step 4: Model Training** - COMPLETE
✅ **Step 5: Model Evaluation** - COMPLETE
✅ **Step 6: Visualization & Deployment** - COMPLETE

---

## Project Structure

```
Sentiment analysis/
├── data/
│   ├── raw/
│   │   └── Dataset-SA.csv               # Original Flipkart dataset
│   └── processed/
│       ├── cleaned_data_step1.csv       # Step 1 output
│       ├── preprocessed_data_step2.csv  # Step 2 output
│       └── ...                          # Intermediate data files
│
├── models/
│   ├── sentiment_classifier_nb.pkl      # Trained Naive Bayes Model
│   └── tfidf_vectorizer.pkl             # TF-IDF Vectorizer
│
├── src/
│   ├── app.py                           # Streamlit Web Application
│   ├── step1_data_collection.py         # Data collection script
│   ├── step2_preprocessing.py           # Preprocessing script
│   ├── step3_feature_extraction.py      # Feature extraction script
│   ├── step4_model_training.py          # Model training script
│   ├── step5_model_evaluation.py        # Model evaluation script
│   └── step6_visualization.py           # Visualization script
│
├── docs/                                # Project documentation
├── outputs/                             # Generated charts and logs
├── requirements.txt                     # Project dependencies
└── README.md                            # This file
```

---

## Key Technologies

- **Python 3.12**
- **Streamlit**: Interactive web application framework
- **Scikit-learn**: Machine learning (Naive Bayes, TF-IDF)
- **pandas & NumPy**: Data manipulation
- **NLTK**: Natural language processing (Tokenization, Lemmatization)
- **Matplotlib & Seaborn**: Data visualization

---

## How to Run Locally

### 1. Prerequisites

Ensure you have Python installed. Install the required dependencies:

```bash
pip install -r requirements.txt
```

### 2. Run the Web Application

To launch the interactive Sentiment Analysis dashboard:

```bash
cd src
streamlit run app.py
```

### 3. Run Pipeline Scripts

You can also run individual steps of the pipeline:

```bash
cd src

# Data Collection
python step1_data_collection.py

# Preprocessing
python step2_preprocessing.py

# Feature Extraction
python step3_feature_extraction.py

# Model Training
python step4_model_training.py

# Model Evaluation
python step5_model_evaluation.py

# Visualization
python step6_visualization.py
```

---

## Project Pipeline Overview

1.  **Data Collection**: Loading and inspecting the 205,052 Flipkart reviews.
2.  **Preprocessing**: Cleaning text (lowercasing, removing stopwords/punctuation, lemmatization).
3.  **Feature Extraction**: Converting text to numerical vectors using TF-IDF (868 features).
4.  **Model Training**: Training a Multinomial Naive Bayes classifier on 80% of the data.
5.  **Model Evaluation**: achieving ~90% accuracy with detailed performance metrics.
6.  **Visualization**: Generating comprehensive charts and a dashboard.
7.  **Deployment**: Interactive web interface for real-time sentiment prediction.

---

## Author

**Project**: Sentiment Analysis on Product Reviews
**Date**: February 2026
**Status**: Completed & Deployed

---

**[Visit the Live App](https://sentiment-projec.streamlit.app/)**
