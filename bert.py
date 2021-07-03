# BERT imports
import os
import torch
from sklearn.feature_extraction.text import CountVectorizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertAdam, BertForSequenceClassification
from tqdm import trange
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef, classification_report
from transformers import AdamW
import pandas as pd

filepath = 'dataset\mtsamples-expanded-preprocessed.csv'
val_filepath = 'dataset/mtsamples-val.csv'

# Set the maximum sequence length.
MAX_LEN = 128
# Select a batch size for training.
batch_size = 15
epochs = 15

PATH = 'bert'

# specify GPU device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
print(torch.cuda.get_device_name(0))


def load_dataset(path):
    print("Loading dataset...")
    df = pd.read_csv(path)
    df = df.dropna()
    return df


def make_labels(df):
    possible_labels = df['medical_specialty'].unique()
    label_dict = {}
    for index, possible_label in enumerate(possible_labels):
        label_dict[possible_label] = index

    df['label'] = df['medical_specialty'].replace(label_dict)


# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def load_df(path):
    df = load_dataset(path)
    keywords = df['keywords'].values
    # LABELS
    make_labels(df)
    y = df['label'].values
    return keywords, y, df


def pad_tokens(tokenizer, tokenized_texts):
    # Pad our input tokens
    # Use the BERT tokenizer to convert the tokens to their index numbers in the BERT vocabulary
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
    return input_ids


def create_att_masks(input_ids):
    # Create attention masks
    attention_masks = []
    # Create a mask of 1s for each token followed by 0s for padding
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)
    return attention_masks


def fine_tune(model):
    # BERT fine-tuning parameters
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]

    # optimizer = BertAdam(optimizer_grouped_parameters,lr=2e-5, warmup=.1)
    optimizer = AdamW(model.parameters(), lr=1e-5)
    return optimizer


def train_BERT(model, optimizer, train_dataloader, train_loss_set):
    # BERT training loop
    for _ in trange(epochs, desc="Epoch"):

        ## TRAINING

        # Set our model to training mode
        model.train()
        # Tracking variables
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        # Train the data for one epoch
        for step, batch in enumerate(train_dataloader):
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch

            # Clear out the gradients (by default they accumulate)
            optimizer.zero_grad()
            # Forward pass

            loss = model(b_input_ids.long(), token_type_ids=None, attention_mask=b_input_mask, labels=b_labels.long())
            train_loss_set.append(loss.item())
            # Backward pass
            loss.backward()
            # Update parameters and take a step using the computed gradient
            optimizer.step()
            # Update tracking variables
            tr_loss += loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1
        print("Train loss: {}".format(tr_loss / nb_tr_steps))

        torch.save(model.state_dict(), os.path.join(PATH, 'bert.pth'))

        return train_loss_set


def validate(model, validation_dataloader):
    ## VALIDATION
    # Put model in evaluation mode
    model.eval()
    # Tracking variables
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    # Evaluate data for one epoch
    for batch in validation_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch
        # Telling the model not to compute or store gradients, saving memory and speeding up validation
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            logits = model(b_input_ids.long(), token_type_ids=None, attention_mask=b_input_mask.long())
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        eval_accuracy += tmp_eval_accuracy
        nb_eval_steps += 1
    print("Validation Accuracy: {}".format(eval_accuracy / nb_eval_steps))


def predict(model, prediction_dataloader):
    ## Prediction on test set
    # Put model in evaluation mode
    model.eval()
    # Tracking variables
    predictions, true_labels = [], []
    # Predict
    for batch in prediction_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch
        # Telling the model not to compute or store gradients, saving memory and speeding up prediction
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            logits = model(b_input_ids.long(), token_type_ids=None, attention_mask=b_input_mask.long())
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        # Store predictions and true labels
        predictions.append(logits)
        true_labels.append(label_ids)

    return predictions, true_labels

def calculate_score(predictions, true_labels):
    matthews_set = []
    for i in range(len(true_labels)):
        matthews = matthews_corrcoef(true_labels[i],
                                     np.argmax(predictions[i], axis=1).flatten())
        matthews_set.append(matthews)

    # Flatten the predictions and true values for aggregate Matthew's evaluation on the whole dataset
    flat_predictions = [item for sublist in predictions for item in sublist]
    flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
    flat_true_labels = [item for sublist in true_labels for item in sublist]

    print('Classification accuracy using BERT Fine Tuning: {0:0.2%}'.format(
        matthews_corrcoef(flat_true_labels, flat_predictions)))


if __name__ == '__main__':
    sentences, labels, df = load_df(filepath)
    nb_labels = df['medical_specialty'].value_counts().count()
    print("categories_num: ", nb_labels)

    # Load BertForSequenceClassification, the pretrained BERT model with a single linear classification layer on top.
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=nb_labels)
    # print(model.cuda())

    # Tokenize with BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    no_model = True;

    try:
        model.load_state_dict(torch.load(os.path.join(PATH, 'bert.pth')))

        model.eval()
        no_model = False
    except:
        print("No model saved.")

    if no_model:
        # queries are stored in the variable query_data_train
        # correct intent labels are stored in the variable labels

        # add special tokens for BERT to work properly
        sentences = ["[CLS] " + query + " [SEP]" for query in sentences]
        print(sentences[0])

        tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
        print("Tokenize the first sentence:")
        print(tokenized_texts[0])

        input_ids = pad_tokens(tokenizer, tokenized_texts)
        attention_masks = create_att_masks(input_ids)

        # Use train_test_split to split our data into train and validation sets for training
        train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels,
                                                                                            random_state=2018,
                                                                                            test_size=0.1)
        train_masks, validation_masks, _, _ = train_test_split(attention_masks, input_ids,
                                                               random_state=2018, test_size=0.1)

        # Convert all of our data into torch tensors, the required datatype for our model
        train_inputs = torch.tensor(train_inputs)
        validation_inputs = torch.tensor(validation_inputs)
        train_labels = torch.tensor(train_labels)
        validation_labels = torch.tensor(validation_labels)
        train_masks = torch.tensor(train_masks)
        validation_masks = torch.tensor(validation_masks)

        # Create an iterator of our data with torch DataLoader
        train_data = TensorDataset(train_inputs, train_masks, train_labels)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
        validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
        validation_sampler = SequentialSampler(validation_data)
        validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

        optimizer = fine_tune(model)

        # Store our loss and accuracy for plotting
        train_loss_set = []

        model.to(device)

        train_loss_set = train_BERT(model, optimizer, train_dataloader, train_loss_set)

        validate(model, validation_dataloader)

        # plot training performance
        plt.figure(figsize=(15, 8))
        plt.title("Training loss")
        plt.xlabel("Batch")
        plt.ylabel("Loss")
        plt.plot(train_loss_set)
        plt.show()

    # TEST
    val_sentences, val_labels, _ = load_df(val_filepath)
    # load test data
    val_sentences = ["[CLS] " + query + " [SEP]" for query in val_sentences]

    tokenized_texts = [tokenizer.tokenize(sent) for sent in val_sentences]
    input_ids = pad_tokens(tokenizer, tokenized_texts)
    attention_masks = create_att_masks(input_ids)

    # create test tensors
    prediction_inputs = torch.tensor(input_ids)
    prediction_masks = torch.tensor(attention_masks)
    prediction_labels = torch.tensor(val_labels)
    batch_size = 32
    prediction_data = TensorDataset(prediction_inputs, prediction_masks, prediction_labels)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

    # Tracking variables
    predictions, true_labels = [], []
    try:
        predictions, true_labels = predict(model, prediction_dataloader)
    except:
        model.to(device)
        predictions, true_labels = predict(model, prediction_dataloader)

    calculate_score(predictions, true_labels)
