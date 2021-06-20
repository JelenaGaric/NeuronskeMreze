import re
import string

import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from keras.layers import Input, Lambda, Dense
from keras.models import Model
import keras.backend as K
import classification
import sys
import spacy
from spacy.lang.en import English
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.preprocessing import LabelEncoder
"""from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
"""
def elmo():
    spacy.load('en_core_web_sm')
    parser = English()

    # np.set_printoptions(threshold=sys.maxsize)

    url = "https://tfhub.dev/google/elmo/2"
    embed = hub.Module(url)

    path = 'dataset\mtsamples-expanded-min.csv'
    path1 = 'dataset\medicalSpeciality1.csv'
    preprocessed_path = 'dataset\dataset_mt.csv'

    # Stop words and special characters
    STOPLIST = set(stopwords.words('english'))
    SYMBOLS = " ".join(string.punctuation).split(" ") + ["-", "...", "”", "”", "''"]


    def plot_word_cloud(text):
        wordcloud_instance = WordCloud(width=800, height=800,
                                       background_color='black',
                                       stopwords=STOPLIST,
                                       min_font_size=10).generate(text)

        plt.figure(figsize=(8, 8), facecolor=None)
        plt.imshow(wordcloud_instance)
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.show()


    try:
        df = pd.read_csv(preprocessed_path, index_col=0)
    except:
        print("No preprocessed dataset.")
        colnames = ["keywords", "medical_specialty", "sample_name", "description", "split"]
        df = pd.read_csv(path, names=colnames, header=None)
        df = df.dropna()
        pd.set_option('max_columns', None)
        preprocessed_keywords = []

        for sentence in df['keywords'].values:
            sentence = re.sub("\S*\d\S*", "", sentence).strip()
            sentence = re.sub('[^A-Za-z]+', ' ', sentence)
            sentence = ' '.join(e.lower() for e in sentence.split() if e.lower() not in classification.stopwords)
            preprocessed_keywords.append(sentence.strip())

        df['preprocessed_keywords'] = preprocessed_keywords
        df.to_csv('dataset\\dataset_mt.csv')

    """texts = ''
    for index, item in df.iterrows():
        texts = texts + ' ' + item['keywords']
    
    plot_word_cloud(texts)"""

    """print(df['medical_specialty'].value_counts())"""
    #df['medical_specialty'] = df['medical_specialty'].apply(classification.remove_spaces)

    labels = set()
    df['medical_specialty'].str.lower().str.split(",").apply(labels.update)
    categories_num = len(labels)
    print("Number of categories: ", categories_num)

    le = LabelEncoder()
    df['medical_specialty_label'] = le.fit_transform(df['medical_specialty'])
    """print(df[["medical_specialty", "medical_specialty_label"]].head(11))"""

    train_df = df.loc[df.split == 'train']
    train_df = train_df.reset_index()
    test_df = df.loc[df.split == 'test']
    test_df = test_df.reset_index()

    # Create datasets (Only take up to 150 words)
    train_text = train_df['preprocessed_keywords'].tolist()
    #train_text = [' '.join(t.split()[0:150]) for t in train_text]
    train_text = np.array(train_text, dtype=object)[:, np.newaxis]

    test_text = test_df['preprocessed_keywords'].tolist()
    #test_text = [' '.join(t.split()[0:150]) for t in test_text]
    test_text = np.array(test_text, dtype=object)[:, np.newaxis]

    """train_label = train_df['medical_specialty_label'].tolist()
    test_label = test_df['medical_specialty_label'].tolist()
    """
    vectorizer = CountVectorizer(tokenizer=lambda x: x.split(","), binary='true')
    train_label = vectorizer.fit_transform(train_df['medical_specialty']).toarray()
    test_label = vectorizer.transform(test_df['medical_specialty']).toarray()

    def ELMoEmbbeding(x):
        return embed(tf.squeeze(tf.cast(x, tf.string), axis=1), signature="default", as_dict=True)["default"]

    input_text = Input(shape=(1,), dtype=tf.string)
    embbeding = Lambda(ELMoEmbbeding, output_shape=(1024,))(input_text)
    dense = Dense(256, activation='relu')(embbeding)
    pred = Dense(38, activation='softmax')(dense) #38
    model = Model(inputs=[input_text], outputs=pred)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train_session():
        with tf.Session() as session:
            K.set_session(session)
            session.run(tf.global_variables_initializer())
            session.run(tf.tables_initializer())
            history = model.fit(train_text, train_label, epochs=5, batch_size=32)
            model.save_weights('./elmo-mt-model.h5')

    def test_session():
        print(test_text)
        with tf.Session() as session:
            K.set_session(session)
            session.run(tf.global_variables_initializer())
            session.run(tf.tables_initializer())
            model.load_weights('./elmo-mt-model.h5')
            predicts = model.predict(test_text, batch_size=2)
            return predicts

    def print_predicts():
        i = 0
        for pred in predicts:
            max_index_col = np.argmax(pred, axis=0)
            print(test_text[i])
            true_labels = []

            labels_indexes = np.where(test_label[i] == 1)
            for l in labels_indexes[0]:
                true_labels.append(list(labels)[l])

            print("Real labels: ", true_labels)
            print("********Prediction*********: ", list(labels)[max_index_col], " (", np.max(pred), ")")

            i = i + 1


    train_session()
    predicts = test_session()
    print_predicts()

    test_clf()

elmo()