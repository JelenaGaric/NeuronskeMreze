import re
import string
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from keras.layers import Input, Lambda, Dense
from keras.models import Model
import keras.backend as K
import spacy
from spacy.lang.en import English
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords

spacy.load('en_core_web_sm')
url = "https://tfhub.dev/google/elmo/3"
embed = hub.Module(url)

path = 'mtsamples-expanded-min.csv'
preprocessed_path = 'mtsamples-preprocessed.csv'

# Stop words and special characters
STOPLIST = set(stopwords.words('english'))
SYMBOLS = " ".join(string.punctuation).split(" ") + ["-", "...", "”", "”", "''"]


def plot_word_cloud(df):
    text = ''
    for index, item in df.iterrows():
        text = text + ' ' + item['keywords']

    wordcloud_instance = WordCloud(width=800, height=800,
                                   background_color='black',
                                   stopwords=STOPLIST,
                                   min_font_size=10).generate(text)

    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud_instance)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()

def remove_categories(df):
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


def ELMoEmbbeding(x):
    return embed(tf.squeeze(tf.cast(x, tf.string), axis=1), signature="default", as_dict=True)["default"]

def make_model():
    input_text = Input(shape=(1,), dtype=tf.string)
    embbeding = Lambda(ELMoEmbbeding, output_shape=(1024,))(input_text)
    dense = Dense(256, activation='relu')(embbeding)
    pred = Dense(categories_num, activation='softmax')(dense)
    model = Model(inputs=[input_text], outputs=pred)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def train_session(model, train_text, train_label, test_text, test_label):
    epochs = 10
    with tf.Session() as session:
        K.set_session(session)
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        history = model.fit(train_text, train_label, epochs=epochs, batch_size=32,
                            validation_data=(test_text, test_label))
        model.save_weights('./elmo-mt-model.h5')

    x = range(epochs)

    plt.plot(x, history.history['accuracy'], label='train')
    plt.plot(x, history.history['val_accuracy'], label='validation')
    plt.title('Accuracy')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()

def test_session(model, test_text):
    with tf.Session() as session:
        K.set_session(session)
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        model.load_weights('./elmo-mt-model.h5')
        predicts = model.predict(test_text, batch_size=2)
        return predicts

def print_predicts(predicts, test_labels, labels):
    i = 0
    for pred in predicts:
        max_index_col = np.argmax(pred, axis=0)
        print(test_text[i])
        true_labels = []

        labels_indexes = np.where(test_labels[i] == 1)
        for l in labels_indexes[0]:
            true_labels.append(list(labels)[l])

        print("Real labels: ", true_labels)
        print("********Prediction*********: ", list(labels)[max_index_col], " (", np.max(pred), ")")

        i = i + 1

def preprocess_text(df):
    preprocessed_keywords = []

    for sentence in df['keywords'].values:
        sentence = re.sub("\S*\d\S*", "", sentence).strip()
        sentence = re.sub('[^A-Za-z]+', ' ', sentence)
        sentence = ' '.join(e.lower() for e in sentence.split() if e.lower() not in STOPLIST)
        preprocessed_keywords.append(sentence.strip())

    df['preprocessed_keywords'] = preprocessed_keywords
    df.to_csv(preprocessed_path)
    return df


if __name__ == '__main__':
    try:
        df = pd.read_csv(preprocessed_path)
    except:
        print("No preprocessed dataset.")
        df = pd.read_csv(path)
        df = remove_categories(df)
        df = preprocess_text(df)

    # plot_word_cloud(texts)

    print(df['medical_specialty'].value_counts())
    labels = set()
    df['medical_specialty'].str.lower().str.split(",").apply(labels.update)
    categories_num = len(labels)
    print("Number of categories: ", categories_num)

    le = LabelEncoder()
    df['medical_specialty_label'] = le.fit_transform(df['medical_specialty'])

    msk = np.random.rand(len(df)) < 0.8
    train_df = df[msk]
    test_df = df[~msk]

    # Create datasets (Only take up to 150 words)
    train_text = train_df['preprocessed_keywords'].tolist()
    train_text = np.array(train_text, dtype=object)[:, np.newaxis]

    test_text = test_df['preprocessed_keywords'].tolist()
    test_text = np.array(test_text, dtype=object)[:, np.newaxis]

    vectorizer = CountVectorizer(tokenizer=lambda x: x.split(","), binary='true')
    train_labels = vectorizer.fit_transform(train_df['medical_specialty']).toarray()
    test_labels = vectorizer.transform(test_df['medical_specialty']).toarray()

    categories_num = df['medical_specialty'].value_counts().count()
    print("categories_num: ", categories_num)

    model = make_model()
    train_session(model, train_text, train_labels, test_text, test_labels)
    # predict a few samples form test dataset
    test_text = test_text[:10]
    predicts = test_session(test_text)
    print_predicts(predicts, test_labels, labels)

