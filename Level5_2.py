import tensorflow as tf 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.layers import Input, Embedding, Dense, Dropout, GlobalAveragePooling1D
from keras.models import Model

# Load the dataset
df = pd.read_csv("C:/Users/Saptadweepa Dutta/Downloads/en-fr-translation-dataset/en-fr.csv", delimiter='\t', header=None, names=['english', 'french'])
df.drop_duplicates(inplace=True)

# Preprocess the data
english_sentences = df['english'].values
french_sentences = df['french'].values

english_tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
english_tokenizer.fit_on_texts(english_sentences)
english_sequences = english_tokenizer.texts_to_sequences(english_sentences)
english_padded_sequences = pad_sequences(english_sequences, maxlen=100, padding='post', truncating='post')

french_tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
french_tokenizer.fit_on_texts(french_sentences)
french_sequences = french_tokenizer.texts_to_sequences(french_sentences)
french_padded_sequences = pad_sequences(french_sequences, maxlen=100, padding='post', truncating='post')

# Create the transformer model
inputs = Input(shape=(100,))
embedding = Embedding(input_dim=5000, output_dim=64)(inputs)
dropout1 = Dropout(0.2)(embedding)
encoder_output, encoder_state = tf.keras.layers.LSTM(64, return_state=True)(dropout1)

decoder_inputs = Input(shape=(None,))
decoder_embedding = Embedding(input_dim=5000, output_dim=64)
decoder_lstm = tf.keras.layers.LSTM(64, return_sequences=True, return_state=True)
decoder_dense = Dense(5000, activation='softmax')

decoder_outputs, _, _ = decoder_lstm(decoder_embedding(decoder_inputs), initial_state=[encoder_output, encoder_state])
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Train the model
english_train, english_test, french_train, french_test = train_test_split(english_padded_sequences,
                                                                          french_padded_sequences, test_size=0.2)

model.fit([english_train, french_train[:, :-1]], french_train[:, 1:], batch_size=64, epochs=20,
          validation_data=([english_test, french_test[:, :-1]], french_test[:, 1:]))

# Define the Flask application
from flask import Flask, request, render_template

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def translate():
    if request.method == 'POST':
        english_sentence = request.form['english']
        english_sequence = english_tokenizer.texts_to_sequences([english_sentence])
        english_padded_sequence = pad_sequences(english_sequence, maxlen=100, padding='post', truncating='post')
        french_padded_sequence = np.zeros((1, 100))
        french_padded_sequence[0, 0] = french_tokenizer.word_index['<start>']

        for i in range(1, 100):
            french_prediction = model.predict([english_padded_sequence, french_padded_sequence])
            french_token_index = np.argmax(french_prediction[0, i - 1, :])
            french_padded_sequence[0, i] = french_token_index

            if french_token_index == french_tokenizer.word_index['<end>']:
                break

        french_sequence = french_padded_sequence[0, 1:]
        french_sentence = french_tokenizer.sequences_to_texts([french_sequence])[0]
        return render_template('index.html', english_sentence=english_sentence, french_sentence=french_sentence)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

