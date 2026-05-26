# 😊 Emotion Classification from Text

## Overview
This project classifies text/sentences into **6 emotion categories** using a **Bidirectional LSTM** neural network. The model is deployed as an interactive **Streamlit web application** for real-time emotion prediction.

## Emotions Predicted
| Emotion | Emoji |
|---------|-------|
| Anger | 😠 |
| Fear | 😨 |
| Joy | 😊 |
| Love | ❤️ |
| Sadness | 😢 |
| Surprise | 😲 |

## Dataset
- Source: [Emotions Dataset for NLP](https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp)
- Training samples: 16,000
- Test samples: 2,000
- Validation samples: 2,000

## Model Architecture
Input Text
↓
Text Preprocessing (Lowercase, Remove punctuation/numbers/URLs, Stopword removal, Lemmatization)
↓
Tokenization (vocab_size = 10,000, max_length = 50)
↓
Embedding Layer (10000 → 128 dimensions)
↓
Bidirectional LSTM (128 units, return_sequences=True)
↓
Dropout (0.3)
↓
Bidirectional LSTM (64 units)
↓
Dropout (0.3)
↓
Dense (64, ReLU activation)
↓
Dense (6, Softmax activation)
↓
Emotion Prediction

## Results

### Classification Report
| Emotion | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| Anger | 0.88 | 0.95 | 0.91 | 275 |
| Fear | 0.93 | 0.83 | 0.88 | 224 |
| Joy | 0.95 | 0.93 | 0.94 | 695 |
| Love | 0.79 | 0.83 | 0.81 | 159 |
| Sadness | 0.97 | 0.96 | 0.96 | 581 |
| Surprise | 0.71 | 0.89 | 0.79 | 66 |

**Overall Accuracy: 92%**


## Tech Stack
- **Framework:** TensorFlow / Keras
- **Model:** Bidirectional LSTM
- **NLP:** NLTK (stopwords, lemmatization)
- **Deployment:** Streamlit
- **Visualization:** Matplotlib, Seaborn
- **Processing:** Pandas, NumPy, Scikit-learn

## Key Features
- ✅ Text preprocessing (lowercase, punctuation removal, stopword removal, lemmatization)
- ✅ Class imbalance handling with `compute_class_weight`
- ✅ Bidirectional LSTM for better context understanding
- ✅ Dropout layers to prevent overfitting
- ✅ Early stopping & learning rate reduction callbacks
- ✅ Interactive Streamlit web app
- ✅ Pre-trained model files included
