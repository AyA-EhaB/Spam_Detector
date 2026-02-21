# 📱 SMS Spam Detector using NLP and Machine Learning

A simple yet effective SMS spam classifier using Natural Language Processing (NLP) techniques and machine learning models. This project is deployed as a web app using Streamlit.

🔗 **Live Demo**: [Spam Detector Web App](https://spamdetector-27z3fba5y5bmdn7r6sgwhu.streamlit.app/)
📂 **Dataset**: [SMS Spam Collection Dataset on Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

---

## 📊 Dataset Overview

* **Name**: SMS Spam Collection
* **Total Messages**: 5,574
* **Classes**: `ham` (legitimate) and `spam`
* Each row contains:

  * `v1`: Label (`ham` or `spam`)
  * `v2`: The SMS message text

## 🧠 Project Workflow

1. **Data Cleaning**:

   * Dropped unnecessary columns
   * Removed special characters, converted text to lowercase
   * Applied stemming and removed stopwords

2. **Feature Extraction**:

   * Used  TF-IDF (Term Frequency - Inverse Document Frequency)

3. **Modeling**:

   * Trained a machine learning model (e.g., Naive Bayes / Logistic Regression/Random Forest)

4. **Evaluation**:

   * Addressed class imbalance
   * Visualized class distribution

5. **Deployment**:

   * Built an interactive web app with **Streamlit**
   * Allows users to enter SMS messages and detect if they are spam or not

---

## 🧪 How to Use the App

1. Visit the [Streamlit App](https://spamdetector-27z3fba5y5bmdn7r6sgwhu.streamlit.app/)
2. Enter an SMS message into the input field
3. Click "Predict"
4. Get instant prediction: `Spam` or `Ham`

---

## Requirements

* `pandas`
* `numpy`
* `sklearn`
* `nltk`
* `streamlit`
* `matplotlib`, `seaborn`

You can install them with:

```bash
pip install -r requirements.txt
```

---

##  Acknowledgement

> Almeida, T.A., Gómez Hidalgo, J.M., Yamakami, A. (2011). *Contributions to the Study of SMS Spam Filtering.*

---
