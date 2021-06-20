import re
from sklearn.metrics import classification_report
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer, AutoTokenizer, AutoModel
import numpy as np
import pandas as pd
from sklearn import svm


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
    #tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    #model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    return model, tokenizer


def get_average_embedding(input_embeddings):
    temp = input_embeddings[0].detach().numpy()
    return np.mean(temp, axis=0)


def get_BERT_embedding(input_seq, **kwargs):
    inputs = tokenizer(input_seq, return_tensors="pt")
    outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state
    if kwargs.get("avg") == 'yes':
        return get_average_embedding(last_hidden_states)
    return last_hidden_states[0][0].detach().numpy()


def make_BERT_embbedings():
    df = pd.read_csv('mtsamples-expanded-min.csv')
    df = df.dropna()
    print(df.shape)
    df = df[df.medical_specialty != 'SOAP / Chart / Progress Notes']
    df = df[df.medical_specialty != 'Consult - History and Phy.']
    df = df[df.medical_specialty != 'Surgery']
    print(df.shape)

    """inputs = tokenizer("Ovo je kosa linija.", return_tensors="pt")
    outputs = model(**inputs)
    last_hidden_state = outputs.last_hidden_state

    print(last_hidden_state.shape)
    print(get_average_embedding(last_hidden_state).shape)
    """
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
            val = np.array(vals)
            x.append(val.astype(np.float))
        else:
            x.append(str_emb)
    return x


def test_clf(clf):
    val_df = pd.read_csv('mtsamples-val.csv')
    val_df = val_df[val_df['keywords'].notna()]
    val_df['keywords'] = val_df['keywords'].apply(clean)
    val_df['embedding'] = val_df.apply(lambda row: get_BERT_embedding(row['keywords']), axis=1)
    val_x = [row for row in val_df['embedding']]

    result = clf.predict(val_x)
    for i in range(len(result)):
        print(val_df.iloc[i]['keywords'][0:15], " ---> ", result[i])

    for index, row in val_df.iterrows():
        val_df.loc[index, 'Prediction'] = result[index]

    del val_df['embedding']
    header = ["NCTId", "keywords", "Prediction"]
    val_df.to_csv("medicalSpecialityBERT.csv", columns=header, index=False)


model, tokenizer = init_BERT()

try:
    df = pd.read_csv('train_BERT_embed.csv')
    print("Loaded dataset with BERT embbedings...")
except:
    print("Making new embbedings...")
    df = make_BERT_embbedings()

clf_svm = svm.SVC(kernel='linear')

"""print(df['medical_specialty'].value_counts())"""
print(df.shape)

msk = np.random.rand(len(df)) < 0.8
train_df = df[msk]
test_df = df[~msk]

train_x = convert_embbedings_to_float(train_df)
test_x = convert_embbedings_to_float(test_df)
train_y = train_df['medical_specialty']
test_y = test_df['medical_specialty']

clf_svm.fit(train_x, train_y)
print(classification_report(test_y, clf_svm.predict(test_x)))

test_clf(clf_svm)