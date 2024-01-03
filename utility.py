import spacy
import re
from transformers import TFBertForSequenceClassification, BertTokenizer
def preprocess_text(text):
    nlp = spacy.load("en_core_web_sm")
    # Lowering down the text
    text = text.lower()

    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Tokenization and lemmatization using spaCy
    doc = nlp(text)
    words = [token.lemma_ for token in doc]

    # Additional stopwords including 'mkr'
    custom_stopwords = set(['rt', '#mkr', "i'm", 'mkr'] + list(nlp.Defaults.stop_words))

    # Removing stopwords
    words = [word for word in words if word not in custom_stopwords]

    # Removing words starting with '@'
    words = [word for word in words if not word.startswith('@')]

    # Removing punctuation and special characters
    words = [word for word in words if word.isalnum()]

    # Join the words back into a string
    processed_text = ' '.join(words)

    return processed_text