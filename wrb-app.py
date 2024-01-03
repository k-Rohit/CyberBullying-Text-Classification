import streamlit as st
import pandas as pd
import tensorflow as tf
from utility import preprocess_text
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import TFBertForSequenceClassification, BertTokenizer

# tokenizer = Tokenizer()
# model_path = 'cyber_model.keras'
# max_seq_length = 23
# loaded_model = tf.keras.models.load_model(model_path)

# Set page title and favicon
st.set_page_config(page_title="Cyberbullying Text Classification", page_icon=":punch:")

# Define the Streamlit app
def main():
    st.title("Cyberbullying Text Classification")

    # # Get user input text
    user_input = st.text_area("Enter text here:")

    # sequence = tokenizer.texts_to_sequences([user_input])

    # #Pad the sequence
    # padded_sequence = pad_sequences(sequence, maxlen=max_seq_length)
    model_name = 'Rohit1234/CYBER-BERT'
    loaded_model = TFBertForSequenceClassification.from_pretrained(model_name)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  # Adjust the tokenizer name as needed
    processed_text = preprocess_text(user_input)
    input_ids = tokenizer.encode(processed_text, truncation=True, padding=True, return_tensors="tf")

    if st.button("Classify"):
        predictions = loaded_model.predict(input_ids)
        predicted_class_index = tf.argmax(predictions.logits, axis=1).numpy()[0]
        st.subheader("Prediction:")
        if predicted_class_index > 0.5:  # Adjust the threshold based on your model
            st.error("This text contains cyberbullying.")
        else:
            st.success("This text is not cyberbullying.")

# Run the app
if __name__ == "__main__":
    main()
