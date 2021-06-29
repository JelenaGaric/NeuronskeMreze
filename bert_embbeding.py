import re
import matplotlib.pyplot as plt
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.models import Sequential
from sklearn.metrics import classification_report
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer, AutoTokenizer, AutoModel
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical


def clean(text):
    REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
    text1 = text.replace('"', ' ')
    text1 = REPLACE_BY_SPACE_RE.sub('', text1)
    text1 = text1.replace('  ', ' ')
    text1 = text1.strip()
    return text1


def init_BERT():
    # Initializing a DistilBERT configuration
    configuration = DistilBertConfig()

    # Initializing a model from the configuration
    model = DistilBertModel(configuration)

    # Accessing the model configuration
    configuration = model.config
    configuration.output_hidden_states = True

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
    model = DistilBertModel.from_pretrained('distilbert-base-multilingual-cased', return_dict=True)
    # tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    # model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    return model, tokenizer


def get_average_embedding(input_embeddings):
    temp = input_embeddings[0].detach().numpy()
    return np.mean(temp, axis=0)


def get_BERT_embedding(input_seq, **kwargs):
    try:
        inputs = tokenizer(input_seq, return_tensors="pt")
        outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        if kwargs.get("avg") == 'yes':
            return get_average_embedding(last_hidden_states)
        return last_hidden_states[0][0].detach().numpy()
    except:
        print("Error")


def make_BERT_embbedings_column():
    df = pd.read_csv('mtsamples-expanded-min.csv')
    df = df.dropna()
    print(df.shape)
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

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
    model = DistilBertModel.from_pretrained('distilbert-base-multilingual-cased', return_dict=True)

    inputs = tokenizer("Ovo je kosa linija.", return_tensors="pt")
    outputs = model(**inputs)
    last_hidden_state = outputs.last_hidden_state

    print(last_hidden_state.shape)
    print(get_average_embedding(last_hidden_state).shape)

    df['embedding'] = df.apply(lambda row: get_BERT_embedding(row['keywords']), axis=1)
    # za operacije na redovima mora da se doda parametar axis=1
    df.to_csv('train_BERT_embed.csv')
    return df


# needed for converting string from loaded df to floats (numpy array)
def convert_embbedings_to_float(df):
    x = []
    for str_emb in df['embedding']:
        if isinstance(str_emb, str):
            str_emb = str_emb.replace('\'', '')
            str_emb = str_emb.replace('[', '')
            str_emb = str_emb.replace(']', '')
            str_emb = str_emb.replace('\n', '')
            vals = str_emb.split()
            # val = list(map(float, vals))
            val = [float(x) for x in vals]
            x.append(val)
        else:
            x.append(str_emb)
    return np.array(x)


def test_clf(clf):
    val_df = pd.read_csv('mtsamples-val.csv')
    val_df = val_df.dropna()
    val_df['keywords'] = val_df['keywords'].apply(clean)
    print(val_df['keywords'])
    val_df['embedding'] = val_df.apply(lambda row: get_BERT_embedding(row['keywords']), axis=1)

    val_x = [row for row in val_df['embedding']]

    result = clf.predict(val_x)

    for i in range(len(result)):
        # print(val_df.iloc[i]['Conditions'], " ---> ", result[i])
        print(val_df.iloc[i]['keywords'], " ---> ", result[i])

    for i in range(len(result)):
        val_df.loc[i, 'Prediction'] = result[i]

    del val_df['embedding']
    # header = ["NCTId", "ShortTitle", "Conditions", "Keywords", "Prediction"]
    header = ["keywords", "medical_specialty", "Prediction"]
    val_df.to_csv("medicalSpecialityBERT.csv", columns=header, index=False)

    val_y = val_df['medical_specialty'].dropna()
    print(classification_report(val_y, result))


model, tokenizer = init_BERT()

try:
    df = pd.read_csv('train_BERT_embed.csv')
    print("Loaded dataset with BERT embbedings...")
except:
    print("Making new embbedings...")
    df = make_BERT_embbedings_column()

# clf_svm = svm.SVC(kernel='linear')
# print(df['medical_specialty'].value_counts())

# labels encoding
target = df['medical_specialty'].values.tolist()
label_encoder = LabelEncoder()
df['medical_specialty_label'] = label_encoder.fit_transform(target)
n_classes = len(label_encoder.classes_)
print(n_classes)

msk = np.random.rand(len(df)) < 0.8
train_df = df[msk]
test_df = df[~msk]

train_x = convert_embbedings_to_float(train_df)
test_x = convert_embbedings_to_float(test_df)
train_y = train_df['medical_specialty_label']
test_y = test_df['medical_specialty_label']
train_labels = train_y.values
test_labels = test_y.values

Y_train = to_categorical(train_labels)
Y_test = to_categorical(test_labels)

print("Before reshape:")
print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)

# reshaping for LSTM
# (NumberOfExamples, TimeSteps, FeaturesPerStep)
# mozda ipak .reshape(train_x.shape[0], train_x.shape[1], 1)
train_x = train_x.reshape(train_x.shape[0], 1, 768)
test_x = test_x.reshape(test_x.shape[0], 1, 768)

print("After reshape:")
print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)

# clf_svm.fit(train_x, train_y)
# print(classification_report(test_y, clf_svm.predict(test_x)))
# test_clf(clf_svm)

# NEURAL NETWORK
model = Sequential()
# define LSTM model
model.add(LSTM(64, return_sequences=False, input_shape=(1, 768)))
# mozda relu
model.add(Dense(256, activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(n_classes, activation='softmax'))
# model.add(Dense(n_classes, activation='sigmoid'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

epochs = 30
hist = model.fit(train_x, Y_train, batch_size=32, epochs=epochs,
                 validation_data=(test_x, Y_test))

x = range(epochs)

plt.plot(x, hist.history['accuracy'], label='train')
plt.plot(x, hist.history['val_accuracy'], label='validation')
plt.title('Accuracy')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()

print((np.expand_dims(test_x[0], axis=0)).shape)
n = 5
predictions = model.predict(test_x[0:n + 1])
max_predictions = []

for i in range(n):
    print(test_df.iloc[i]["keywords"])
    max_predictions.append(np.argmax(predictions[i], axis=0))

print(max_predictions)

print(label_encoder.inverse_transform(max_predictions))