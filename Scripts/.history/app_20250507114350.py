import streamlit as st
import pickle
import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import nltk
import pandas as pd
import numpy as np
import time

# Set page configuration
st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="📱",
    layout="centered"
)

# Apply custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.3rem;
        color: #424242;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .spam-result {
        padding: 20px;
        border-radius: 10px;
        font-weight: bold;
        text-align: center;
        font-size: 1.5rem;
        margin: 20px 0;
    }
    .spam {
        background-color: #ffcdd2;
        color: #c62828;
        border: 2px solid #c62828;
    }
    .ham {
        background-color: #c8e6c9;
        color: #2e7d32;
        border: 2px solid #2e7d32;
    }
    .model-metrics {
        margin-top: 10px;
        padding: 10px;
        background-color: #f5f5f5;
        border-radius: 5px;
    }
    .stTextInput > div > div > input {
        font-size: 1.1rem;
        padding: 15px 10px;
    }
    .stButton>button {
        background-color: #1E88E5;
        color: white;
        font-weight: bold;
        padding: 0.5rem 1rem;
        font-size: 1.1rem;
        border-radius: 5px;
        border: none;
        width: 100%;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        color: #757575;
        font-size: 0.8rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Display header
st.markdown('<h1 class="main-header">SMS Spam Detector</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Check if your SMS message is spam or legitimate</p>', unsafe_allow_html=True)

# Download NLTK resources
@st.cache_resource
def download_nltk_resources():
    nltk.download('stopwords')
    return PorterStemmer()

Stemmer = download_nltk_resources()

# Text preprocessing function
def preprocess(text):
    if not text:
        return ""
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [Stemmer.stem(word) for word in text if not word in stopwords.words('english')]
    text = ' '.join(text)
    return text

# Load models
@st.cache_resource
def load_models():
    try:
        import os
        
        # Get the current script's directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Navigate to the project root (one level up from Scripts)
        project_root = os.path.dirname(current_dir)
        
        # Path to the Artifacts folder
        artifacts_dir = os.path.join(project_root, "Artifacts")
        
        # Load the models with the correct paths
        tfidf_path = os.path.join(artifacts_dir, 'tfidf.pkl')
        rf_path = os.path.join(artifacts_dir, 'rf.pkl')
        lr_path = os.path.join(artifacts_dir, 'lr.pkl')
        
        tfidf = pickle.load(open(tfidf_path, 'rb'))
        rf = pickle.load(open(rf_path, 'rb'))
        lr = pickle.load(open(lr_path, 'rb'))
        
        return tfidf, rf, lr
    except FileNotFoundError as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

tfidf, rf, lr = load_models()

# Function to get confidence scores from models
def get_confidence(model, data):
    probabilities = model.predict_proba(data)
    return probabilities[0][1]  # Probability of class 1 (spam)

# Get example messages


# Create tabs for input and examples
tab1, tab2 = st.tabs(["📝 Check Your Message", "📚 Example Messages"])

with tab1:
    # User input
    msg = st.text_area("Enter the message to analyze:", height=150, 
                       placeholder="Type or paste your SMS message here...")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        analyze_button = st.button("Analyze Message", use_container_width=True)
    with col2:
        clear_button = st.button("Clear Input", use_container_width=True)
    
    if clear_button:
        msg = ""
        st.experimental_rerun()
    
    # Process prediction when button is clicked
    if analyze_button and msg:
        with st.spinner('Analyzing message...'):
            # Add a slight delay for better UX
            time.sleep(0.5)
            
            # Preprocess text
            preprocessed = preprocess(msg)
            
            if not preprocessed:
                st.warning("Please enter a valid message with text content.")
            else:
                # Transform using TF-IDF
                data = tfidf.transform([preprocessed]).toarray()
                
                # Get predictions
                lr_pred = lr.predict(data)[0]
                rf_pred = rf.predict(data)[0]
                
                # Get confidence scores
                lr_confidence = get_confidence(lr, data) * 100
                rf_confidence = get_confidence(rf, data) * 100
                
                # Display results
                st.markdown("### Analysis Results")
                
                # Create columns for the two models
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Logistic Regression")
                    prediction_text = "SPAM" if lr_pred else "HAM (Legitimate)"
                    prediction_class = "spam" if lr_pred else "ham"
                    st.markdown(f'<div class="spam-result {prediction_class}">{prediction_text}</div>', unsafe_allow_html=True)
                    # Convert percentage (0-100) to fraction (0.0-1.0) for progress bar
                    progress_value = (lr_confidence / 100) if lr_pred else (1 - lr_confidence / 100)
                    st.progress(progress_value)
                    st.write(f"Confidence: {lr_confidence:.1f}%" if lr_pred else f"Confidence: {100 - lr_confidence:.1f}%")
                
                with col2:
                    st.markdown("#### Random Forest")
                    prediction_text = "SPAM" if rf_pred else "HAM (Legitimate)"
                    prediction_class = "spam" if rf_pred else "ham"
                    st.markdown(f'<div class="spam-result {prediction_class}">{prediction_text}</div>', unsafe_allow_html=True)
                    # Convert percentage (0-100) to fraction (0.0-1.0) for progress bar
                    progress_value = (rf_confidence / 100) if rf_pred else (1 - rf_confidence / 100)
                    st.progress(progress_value)
                    st.write(f"Confidence: {rf_confidence:.1f}%" if rf_pred else f"Confidence: {100 - rf_confidence:.1f}%")
                
                # Overall recommendation
                st.markdown("### Final Verdict")
                if rf_pred and lr_pred:
                    st.error("⚠️ Both models identify this as SPAM. Be very cautious!")
                elif rf_pred or lr_pred:
                    st.warning("⚠️ One model flagged this as potential SPAM. Be cautious.")
                else:
                    st.success("✅ This message appears to be legitimate.")
                    
                # Show preprocessing
                with st.expander("View preprocessing details"):
                    st.markdown("#### Original Message")
                    st.write(msg)
                    st.markdown("#### Preprocessed Text")
                    st.write(preprocessed if preprocessed else "(No text content after preprocessing)")

# Information about the model
with st.expander("About this Spam Detector"):
    st.markdown("""
    ### SMS Spam Detector
    
    This application uses Machine Learning to classify SMS messages as either 'spam' or 'ham' (legitimate).
    
    #### How it works:
    1. Your message is preprocessed (removing special characters, converting to lowercase, etc.)
    2. Text is transformed into a numerical format using TF-IDF vectorization
    3. Two different machine learning models analyze the message:
       - Logistic Regression
       - Random Forest
       
    #### Dataset:
    This model was trained on the SMS Spam Collection Dataset from Kaggle, containing 5,574 SMS messages.
    """)

st.markdown('<div class="footer">Created with Streamlit • SMS Spam Detection</div>', unsafe_allow_html=True)