import nltk
import pandas as pd
import re
import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import Embedding, LeakyReLU, Conv1D, MaxPooling1D, LSTM, Bidirectional, Dropout, Dense, Flatten
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow import keras
from gensim.models import Word2Vec
import numpy as np
from nltk.corpus import stopwords
from classification import lemmatize_sentence

path = 'mtsamples-expanded-min.csv'
preprocessed_path = 'mtsamples-val.csv'

stopwords = set(stopwords.words('english'))

def import_dataset(path):
    df = pd.read_csv(path)
    df = df.dropna()
    df = df[df.medical_specialty != 'SOAP / Chart / Progress Notes']
    df = df[df.medical_specialty != 'Consult - History and Phy.']
    df = df[df.medical_specialty != 'Surgery']
    df = df[df.medical_specialty != 'Sleep Medicine']
    df = df[df.medical_specialty != 'metabolics / Gastroenterology']
    df = df[df.medical_specialty != 'Office Notes']
    df = df[df.medical_specialty != 'Lab Medicine - Pathology']
    df = df[df.medical_specialty != 'IME-QME-Work Comp etc.']
    df = df[df.medical_specialty != 'Hospice - Palliative Care']
    df = df[df.medical_specialty != 'Emergency Room Reports']
    df = df[df.medical_specialty != 'Discharge Summary']
    df = df[df.medical_specialty != 'Diets and Nutritions']
    df = df[df.medical_specialty != 'Cosmetic / Plastic Surgery']
    df = df[df.medical_specialty != 'Chiropractic']
    df = df[df.medical_specialty != 'Letters']
    df = df[df.medical_specialty != 'Physical Medicine - Rehab']
    df = df[df.medical_specialty != 'Speech - Language']
    return df

def preprocess_text(df, save):
    preprocessed_keywords = []

    for sentence in df['keywords'].values:
        sentence = decontracted(sentence)
        sentence = re.sub("\S*\d\S*", "", sentence).strip()
        sentence = re.sub('[^A-Za-z]+', ' ', sentence)
        sentence = ' '.join(e.lower() for e in sentence.split() if e.lower() not in stopwords)
        sentence = lemmatize_sentence(sentence)
        preprocessed_keywords.append(sentence.strip())

    df['preprocessed_keywords'] = preprocessed_keywords
    if save:
        df.to_csv(preprocessed_path)
    return df

def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase


def encode(df):
    vect=Tokenizer()
    vect.fit_on_texts(df['preprocessed_keywords'])
    vocab_size = len(vect.word_index) + 1
    print("Vocab size: ", vocab_size)

    encoded_docs = vect.texts_to_sequences(df['preprocessed_keywords'])
    padded_docs = pad_sequences(encoded_docs, maxlen=100, padding='post')
    return vect, padded_docs, vocab_size

def create_model(categories_num, embedding_vector_length, embedding_matrix):
    # create the model
    nn = Sequential()
    nn.add(Embedding(input_dim=vocab_size,
                     output_dim=embedding_vector_length,
                     embeddings_initializer=keras.initializers.Constant(embedding_matrix),
                     trainable=False,
                     input_length=100))
    # TODO: Maybe try LeakyRELU activation on Conv1D instead of relu
    # TODO: more Conv1Ds
    nn.add(Conv1D(filters=32, kernel_size=4, padding='same', activation='relu'))
    nn.add(MaxPooling1D(pool_size=2, strides=1))
    # define LSTM model
    # TODO: Maybe add tanh activation on one of LSTM layers
    nn.add(Bidirectional(LSTM(1024, return_sequences=True)))
    nn.add(Dropout(0.5))
    # Adding a dropout layer
    # TODO: Less dropout and LSTM with more inputs
    # TODO: Bidirectional LSTM
    #nn.add(LSTM(128))
    #nn.add(Dropout(0.1))
    # TODO: Maybe one more Dense layer with more neurons (often when you don't have enough neurons in the layer before the output layer, the model loses accuracy)
    nn.add(Flatten())
    # Adding a dense output layer with sigmoid activation
    nn.add(Dense(categories_num, activation='sigmoid'))

    print(nn.summary())

    nn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return nn


def train(model, x_train, y_train, x_test, y_test):
    epochs = 20
    history = model.fit(x_train, y_train,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_test, y_test),
                        batch_size=16)
    #TODO: try with different batch sizes

    model.save('cnn_lstm_model')

    x = range(epochs)

    plt.plot(x, history.history['acc'], label='train')
    plt.plot(x, history.history['val_acc'], label='validation')
    plt.title('Accuracy')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()

# function to return key for any value
def get_key(val, dict):
    for key, value in dict.items():
        if val == value:
            return key

    return "key doesn't exist"

def predict_sample():

    val_df = pd.read_csv('dataset/mtsamples-val.csv')
    val_df = val_df[val_df['keywords'].notna()]
    val_df = preprocess_text(val_df, save=False)

    t = val_df.sample(1)
    encoded_docs = vect.texts_to_sequences(t['preprocessed_keywords'])
    padded_docs = pad_sequences(encoded_docs, maxlen=100, padding='post')

    pred = model.predict(padded_docs)
    print(pred)
    max_index = np.argmax(pred[0], axis=0)
    predicted_specialty = get_key(max_index, vectorizer.vocabulary_)

    print("keywords -->", t['keywords'].values)
    print("medical_specialty -->", t['medical_specialty'].values)
    print("Predicted tags -->", predicted_specialty)

try:
    df = import_dataset(preprocessed_path)
except:
    print("No preprocessed dataset.")
    df = import_dataset(path)
    df = preprocess_text(df, save=True)

all_words = [nltk.word_tokenize(sent) for sent in df["preprocessed_keywords"]]
word2vec = Word2Vec(all_words, min_count=2)
vocabulary = word2vec.wv.key_to_index

v1 = word2vec.wv['biopsy']
sim_words = word2vec.wv.most_similar('biopsy')
print(sim_words)

msk = np.random.rand(len(df)) < 0.8
train_df = df[msk]
test_df = df[~msk]

vect, padded_docs_train, vocab_size = encode(train_df)
_, padded_docs_test, _ = encode(test_df)

embedding_vector_length = 100
hits = 0
misses = 0

# Prepare embedding matrix
embedding_matrix = np.zeros((vocab_size, embedding_vector_length))
for word, i in vocabulary.items():
    embedding_vector = word2vec.wv[word]
    if embedding_vector is not None:
        # Words not found in embedding index will be all-zeros.
        # This includes the representation for "padding" and "OOV"
        embedding_matrix[i] = embedding_vector
        hits += 1
    else:
        misses += 1
print("Converted %d words (%d misses)" % (hits, misses))

print(df['medical_specialty'].value_counts())
categories_num = df['medical_specialty'].value_counts().count()
print("categories_num: ", categories_num)

vectorizer = CountVectorizer(tokenizer=lambda x: x.split(","), binary='true')
vectorizer.fit(df['medical_specialty'])
y_train = vectorizer.transform(train_df['medical_specialty']).toarray()
y_test = vectorizer.transform(test_df['medical_specialty']).toarray()

#if not os.path.exists('./cnn_lstm_model'):
print("No model saved. Training a new one.")
model = create_model(categories_num, embedding_vector_length, embedding_matrix);
train(model, padded_docs_train, y_train, padded_docs_test, y_test)
"""else:
    reconstructed_model = keras.models.load_model("cnn_lstm_model")
    model = reconstructed_model
    print("Loaded model.")"""


predict_sample()
predict_sample()
predict_sample()
