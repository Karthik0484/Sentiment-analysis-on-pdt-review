# Sentiment Analysis on Product Reviews

A complete data science project for analyzing customer sentiment from Flipkart product reviews.

## Project Status

✅ **Step 1: Data Collection** - COMPLETE
✅ **Step 2: Data Cleaning & Preprocessing** - COMPLETE
⏳ **Step 3: Feature Extraction** - Pending
⏳ **Step 4: Model Training** - Pending
⏳ **Step 5: Model Evaluation** - Pending

---

## Project Structure

```
Sentiment analysis/
├── data/
│   ├── raw/
│   │   └── Dataset-SA.csv              # Original Flipkart dataset (205,052 reviews)
│   └── processed/
│       ├── cleaned_data_step1.csv       # Step 1 output (180,388 clean reviews)
│       ├── preprocessed_data_step2.csv  # Step 2 output (main file)
│       └── preprocessed_data_step2_full.csv  # Step 2 output (with all intermediate steps)
│
├── src/
│   ├── step1_data_collection.py         # Step 1: Data collection script
│   └── step2_preprocessing.py           # Step 2: Preprocessing script
│
├── docs/
│   ├── STEP1_DOCUMENTATION.md           # Step 1 technical documentation
│   ├── STEP2_DOCUMENTATION.md           # Step 2 technical documentation
│   ├── VIVA_EXPLANATION_STEP1.md        # Step 1 viva preparation
│   └── VIVA_EXPLANATION_STEP2.md        # Step 2 viva preparation
│
├── outputs/
│   └── output.txt                       # Script execution logs
│
└── README.md                            # This file
```

---

## Dataset Information

**Source**: Flipkart Product Reviews
- **Original records**: 205,052 customer reviews
- **After cleaning**: 180,388 reviews (87.97% retention)
- **Columns**: review_text, rating (1-5 stars)

**Rating Distribution**:
- 5 stars: 105,647 reviews (58.57%)
- 4 stars: 36,969 reviews (20.49%)
- 3 stars: 14,024 reviews (7.77%)
- 2 stars: 5,451 reviews (3.02%)
- 1 star: 18,294 reviews (10.14%)

---

## How to Run

### Prerequisites

```bash
pip install pandas numpy nltk
```

### Step 1: Data Collection

```bash
cd src
python step1_data_collection.py
```

**What it does**:
1. Loads the raw dataset from `data/raw/Dataset-SA.csv`
2. Explores and analyzes the data structure
3. Identifies relevant columns for sentiment analysis
4. Checks for missing values
5. Selects review text and rating columns
6. Removes rows with missing reviews
7. Saves cleaned data to `data/processed/cleaned_data_step1.csv`

**Output**: 180,388 clean reviews with review_text and rating

---

### Step 2: Data Cleaning & Preprocessing

```bash
cd src
python step2_preprocessing.py
```

**What it does**:
1. Loads cleaned data from Step 1
2. Handles missing values (verification)
3. Converts text to lowercase
4. Removes punctuation, numbers, and special characters
5. Removes extra whitespaces
6. Removes 198 English stopwords
7. Tokenizes text into words
8. Applies lemmatization using WordNet
9. Creates `cleaned_text` column (preserves original)
10. Saves preprocessed data

**Output**: 
- `preprocessed_data_step2.csv` (3 columns: review_text, rating, cleaned_text)
- `preprocessed_data_step2_full.csv` (with all intermediate steps)

---

## Preprocessing Details

### Text Transformation Example

```
ORIGINAL: "This is a GREAT product! Worth the money."
          ↓ Lowercase
          "this is a great product! worth the money."
          ↓ Remove punctuation
          "this is a great product worth the money"
          ↓ Remove stopwords (this, is, a, the)
          "great product worth money"
          ↓ Tokenize
          ['great', 'product', 'worth', 'money']
          ↓ Lemmatize
          ['great', 'product', 'worth', 'money']
CLEANED:  "great product worth money"
```

### Statistics

- **Average word reduction**: 15.59%
- **Average tokens per review**: 1.60
- **Stopwords removed**: 198 common English words
- **Vocabulary reduction**: ~60-70%

---

## Documentation

All documentation is available in the `docs/` directory:

### Technical Documentation
- **STEP1_DOCUMENTATION.md**: Complete technical details for Step 1
- **STEP2_DOCUMENTATION.md**: Complete technical details for Step 2

### Viva/Project Review Preparation
- **VIVA_EXPLANATION_STEP1.md**: Q&A and explanations for Step 1
- **VIVA_EXPLANATION_STEP2.md**: Q&A and explanations for Step 2

---

## Key Technologies

- **Python 3.12**
- **pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **NLTK**: Natural language processing
  - Stopwords corpus
  - Word tokenization
  - WordNet lemmatization

---

## Results Summary

### Step 1: Data Collection
✓ Loaded 205,052 reviews
✓ Identified meaningful columns
✓ Removed 24,664 rows with missing reviews (12.03%)
✓ Final clean dataset: 180,388 reviews

### Step 2: Preprocessing
✓ All 8 preprocessing steps completed
✓ Text normalized and standardized
✓ 15.59% word reduction (noise removed)
✓ Original text preserved for reference
✓ Ready for feature extraction

---

## Next Steps

### Step 3: Feature Extraction (Upcoming)
- TF-IDF vectorization
- Bag of Words representation
- Word embeddings (Word2Vec/GloVe)
- N-gram analysis

### Step 4: Model Training (Upcoming)
- Train-test split (80-20)
- Naive Bayes classifier
- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest

### Step 5: Model Evaluation (Upcoming)
- Accuracy, Precision, Recall, F1-score
- Confusion matrix
- ROC curve and AUC
- Cross-validation

---

## Author

**Project**: Sentiment Analysis on Product Reviews
**Date**: February 2026
**Status**: Steps 1-2 Complete

---

## Notes

- All scripts use relative paths and work from the `src/` directory
- Original data is preserved at every step
- Comprehensive documentation provided for viva/project review
- Code is production-ready with proper error handling

---

## Quick Start

```bash
# Navigate to src directory
cd "c:\Users\KARTHIK\OneDrive\Desktop\Sentiment analysis\src"

# Run Step 1
python step1_data_collection.py

# Run Step 2
python step2_preprocessing.py

# Check outputs in data/processed/
```

---

**Project successfully restructured and operational!** ✅
