"""
Sentiment Analysis App
User Interface for Product Review Sentiment Prediction
Optimized for Performance (Caching & Latency Reduction)
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import re
import nltk

# ============================================================================
# PERFORMANCE OPTIMIZATION: 1. SETUP & CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Sentiment Analysis System",
    page_icon="🤖",
    layout="centered"
)

# ============================================================================
# PERFORMANCE OPTIMIZATION: 2. CACHED RESOURCE LOADING
# ============================================================================

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')

@st.cache_resource(show_spinner=False)
def initialize_nltk():
    """
    Optimize NLTK usage by downloading resources only once
    and returning necessary objects to avoid re-initialization.
    """
    # Define resources and their specific lookup paths to prevent false negatives
    # Format: (resource_name, lookup_path)
    resources = [
        ('punkt', 'tokenizers/punkt'),
        ('punkt_tab', 'tokenizers/punkt_tab'),
        ('stopwords', 'corpora/stopwords'),
        ('wordnet', 'corpora/wordnet'),
        ('omw-1.4', 'corpora/omw-1.4')
    ]
    
    for resource, path in resources:
        try:
            # Check if the resource already exists
            nltk.data.find(path)
        except LookupError:
            # If not found, download it silently
            nltk.download(resource, quiet=True)
    
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    
    # Pre-load stopwords set to avoid rebuilding it every time
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    return word_tokenize, stop_words, lemmatizer

# Initialize NLTK resources once
word_tokenize, stop_words, lemmatizer = initialize_nltk()

@st.cache_resource(show_spinner=False)
def load_models():
    """
    Load ML models and vectorizers only once and cache them in memory.
    This prevents reloading large files on every interaction.
    """
    try:
        # Load Vectorizer
        vectorizer_path = os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl')
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
            
        # Load Model
        model_path = os.path.join(MODELS_DIR, 'sentiment_classifier_nb.pkl')
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            
        return vectorizer, model
    except FileNotFoundError:
        return None, None

# Load models with a spinner only on the first run
with st.spinner("🚀 Booting up AI Engine..."):
    vectorizer, model = load_models()

if vectorizer is None or model is None:
    st.error("❌ Model files not found! Please ensure Step 3 and Step 4 are completed.")
    st.stop()

# ============================================================================
# PERFORMANCE OPTIMIZATION: 3. CACHED PREPROCESSING
# ============================================================================

@st.cache_data(show_spinner=False)
def preprocess_text(text):
    """
    Cache the preprocessing results.
    If the user analyzes the same text again, we fetch the result instantly
    instead of re-running regex, tokenization, and lemmatization.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
        
    # 1. Lowercasing
    text = text.lower()
    
    # 2. Remove punctuation, special characters, and numbers
    # Combined regex for speed
    text = re.sub(r'[^\w\s]|[\d]', '', text)
    
    # 3. Tokenization (using cached function)
    tokens = word_tokenize(text)
    
    # 4. Stopwords Removal & Lemmatization (using cached objects)
    # List comprehension for speed
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
    
    return ' '.join(tokens)

# ============================================================================
# USER INTERFACE
# ============================================================================

# Header
st.title("🛍️ Sentiment Analysis System")
st.markdown("Analyze product reviews and detect sentiment instantly using AI.")
st.markdown("---")

# Main Input Section
col1, col2 = st.columns([2, 1])

with col1:
    product_name = st.text_input(
        "Product Name (Optional)",
        placeholder="e.g., Samsung Galaxy M14"
    )

with col2:
    rating = st.selectbox(
        "User Rating (Optional)",
        options=[1, 2, 3, 4, 5],
        index=4
    )

review_type = st.radio(
    "Review Format",
    options=["Text Review", "Audio Review (Coming Soon)", "Video Review (Coming Soon)"],
    index=0,
    horizontal=True
)

if review_type != "Text Review":
    st.info("⚠️ This feature is currently under development. Using Text Review mode.")

review_text = st.text_area(
    "Review Text",
    height=150,
    placeholder="Type your detailed product review here..."
)

# 5. Submit Button
analyze_btn = st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True)

# ============================================================================
# PROCESSING & RESULTS
# ============================================================================

if analyze_btn:
    if not review_text.strip():
        st.warning("⚠️ Please provide some review text to analyze.")
    else:
        # Use a spinner for perceived performance
        with st.spinner("Analyzing..."):
            # 1. Preprocess (Cached)
            processed_text = preprocess_text(review_text)
            
            # 2. Vectorize & Predict
            # Transform returns a sparse matrix; we keep it sparse for efficiency
            features = vectorizer.transform([processed_text])
            
            # 3. Predict Probability
            probabilities = model.predict_proba(features)[0]
            
            # 4. Predict Class
            prediction_idx = np.argmax(probabilities)
            prediction_label = model.classes_[prediction_idx]
            confidence = probabilities[prediction_idx] * 100
            
            # Map sentiment details
            sentiment_map = {
                'positive': {'color': 'green', 'emoji': '😃', 'msg': 'Positive Feedback'},
                'neutral': {'color': 'orange', 'emoji': '😐', 'msg': 'Neutral Feedback'},
                'negative': {'color': 'red', 'emoji': '😔', 'msg': 'Negative Feedback'}
            }
            
            result = sentiment_map.get(prediction_label, {'color': 'gray', 'emoji': '❓'})
            
            # Display Results
            st.markdown("### Analysis Results")
            
            st.markdown(
                f"""
                <div style="
                    background-color: #f0f2f6; 
                    padding: 20px; 
                    border-radius: 10px; 
                    border-left: 5px solid {result['color']};
                    text-align: center;">
                    <h2 style="color: {result['color']}; margin:0;">
                        {result['emoji']} {prediction_label.capitalize()}
                    </h2>
                    <p style="margin:5px; font-size: 18px;">
                        Confidence: <strong>{confidence:.2f}%</strong>
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            if rating:
                st.markdown("") 
                is_match = (
                    (rating >= 4 and prediction_label == 'positive') or 
                    (rating == 3 and prediction_label == 'neutral') or 
                    (rating <= 2 and prediction_label == 'negative')
                )
                
                if is_match:
                    st.success("✅ Prediction matches the user rating!")
                else:
                    st.caption("ℹ️ Note: The sentiment detected differs from the star rating.")

            with st.expander("See Detailed Confidence Scores"):
                probs_df = pd.DataFrame({
                    'Sentiment': [c.capitalize() for c in model.classes_],
                    'Probability': probabilities
                })
                st.bar_chart(probs_df.set_index('Sentiment'))

st.markdown("---")
st.caption("Sentiment Analysis Project | Optimized with Streamlit Caching")
