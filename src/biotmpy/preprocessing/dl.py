
# Imports
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm
import os
import pandas as pd
import random
import pickle
import numpy as np
import nltk
from nltk.corpus import stopwords
import string
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_auc_score, roc_curve, precision_recall_curve


def train_val_split(x_train, y_train, validation_percentage=25, seed_value = None, abstract_set=None, model=None):
    """
    Splits the training set (x_train and y_train) into a training and validation sets. The validation set is used to evaluate the model during training.

    :param x_train: training set
    :param y_train: training set labels
    :param validation_percentage: percentage of the training set to be used as validation set. Default is 25%.
    :param seed_value: seed value for the random number generator. Default is None.
    :param abstract_set: set of abstracts. Default is None.
    :param model: model to be used. Default is None.

    :return: shorter_x_train, shorter_y_train, x_val, y_val
    """
    if model == 'bert':
        return train_val_split_bert(x_train, y_train, validation_percentage, seed_value, abstract_set, model)

    else:
        validation_samples = int(x_train.shape[0] * (validation_percentag e /100))
        training_samples = int(x_train.shape[0] - validation_samples)

        if seed_value:
            np.random.seed(seed_value)

        indices = np.arange(x_train.shape[0])

        np.random.shuffle(indices)

        x_train = x_train[indices]

        y_train = y_train[indices]

        shorter_x_train = x_train[:training_samples]
        shorter_y_train = y_train[:training_samples]
        
        x_val = x_train[training_samples: training_samples + validation_samples]
        y_val = y_train[training_samples: training_samples + validation_samples]

        return shorter_x_train, shorter_y_train, x_val, y_val


def train_val_split_bert(x_train, y_train, validation_percentage=25, seed_value=None, abstract_set=None,
                         model=None):
    """
    Splits the training set (x_train and y_train) into a training and validation sets. The validation set is used to evaluate the model during training. This function is used for BERT models. BERT models require a different input format than other models.

    :param x_train: training set
    :param y_train: training set labels
    :param validation_percentage: percentage of the training set to be used as validation set. Default is 25%.
    :param seed_value: seed value for the random number generator. Default is None.
    :param abstract_set: set of abstracts. Default is None.
    :param model: model to be used. Default is None.

    :return: shorter_x_train, shorter_y_train, x_val, y_val
    """
    validation_samples = int(x_train[0].shape[0] * (validation_percentag e /100))
    training_samples = int(x_train[0].shape[0] - validation_samples)

    if seed_value:
        np.random.seed(seed_value)

    indices = np.arange(x_train.shape[0])

    np.random.shuffle(indices)

    for i_array in range(len(x_train)):
        x_train[i_array] = x_train[i_array][indices]
        
    y_train = y_train[indices]

    shorter_x_train = []
    for i_array in range(len(x_train)):
        shorter_x_train.append(x_train[i_array][:training_samples])    

    shorter_y_train = y_train[:training_samples]

    x_val = []
    for i_array in range(len(x_train)):
        x_val.append(x_train[i_array][training_samples: training_samples + validation_samples])

    y_val = y_train[training_samples: training_samples + validation_samples]
    
    return shorter_x_train, shorter_y_train, x_val, y_val
def DL_preprocessing(x_set, y_set, config, dataset, validation_percentage=None, seed_value=None):  # model only needed for DeepDTA (model='DeepDTA')
    """
    Preprocesses the data for deep learning models.

    :param x_set: set of documents
    :param y_set: set of labels
    :param config: configuration for deep learning models
    :param dataset: dataset to be used. Can be 'train' or 'test'.
    """
    padding = config.padding
    truncating = config.truncating
    oov_token = config.oov_token
    lower = config.lower
    remove_punctuation = config.remove_punctuation

    texts, word_index = [], []
    title_texts, abstract_texts = [], []
    docs = list(x_set['Document'])
    for doc in docs:
        texts.append(doc.fulltext_string)
        # if model == 'DeepDTA':
        #     title_texts.append(doc.title_string)
        #     abstract_texts.append(doc.abstract_string)


    if dataset == 'train':
        config.tokenizer.fit_on_texts(texts)
        word_index = config.tokenizer.word_index
        print('Found %s unique tokens.' % len(word_index))

    max_sent_len = config.max_sent_len

    if config.max_nb_sentences:
        max_nb_sentences = config.max_nb_sentences
        x_set = np.zeros((len(texts), max_nb_sentences, max_sent_len), dtype='int32')
        for i, doc in enumerate(docs):
            for j, sent in enumerate(doc.sentences):
                if j < max_nb_sentences:
                    wordTokens = text_to_word_sequence(sent.string, lower=lower)
                    if not remove_punctuation:
                        wordTokens = text_to_word_sequence(sent.string, lower=lower, filters='')
                    k = 0
                    pre_truncating_start = len(wordTokens ) -max_sent_len
                    for id_word, word in enumerate(wordTokens):
                        if truncating == 'pre' and len(wordTokens) > max_sent_len:
                            if id_word < pre_truncating_start:
                                continue
                        if k < max_sent_len:
                            if word in config.tokenizer.word_index:
                                if config.tokenizer.word_index[word] < config.max_nb_words:
                                    if paddin g= ='pre' and len(wordTokens) < max_sent_len:
                                        x_set[i, j, k+ (max_sent_len - len(wordTokens))] = \
                                        config.tokenizer.word_index[word]
                                    else:
                                        x_set[i, j, k] = config.tokenizer.word_index[word]
                                    k = k + 1
                            else:
                                if padding == 'pre' and len(wordTokens) < max_sent_len:
                                    x_set[i, j, k + (max_sent_len - len(wordTokens))] = config.tokenizer.word_index[
                                        str(oov_token)]
                                else:
                                    x_set[i, j, k] = config.tokenizer.word_index[str(oov_token)]
                                k += 1
        if oov_token:
            print('Index of Unknown Words:', config.tokenizer.word_index[str(oov_token)])
    else:

        if model == 'DeepDTA':
            title_sequences = config.tokenizer.texts_to_sequences(title_texts)
            title_set = pad_sequences(title_sequences, maxlen=max_sent_len, padding=padding, truncating=truncating)
            abstract_sequences = config.tokenizer.texts_to_sequences(abstract_texts)
            abstract_set = pad_sequences(abstract_sequences, maxlen=max_sent_len, padding=padding,
                                         truncating=truncating)
        else:
            sequences = config.tokenizer.texts_to_sequences(texts)
            x_set = pad_sequences(sequences, maxlen=max_sent_len, padding=padding, truncating=truncating)

    y_set = np.asarray(y_set)

    if not validation_percentage or validation_percentage == 0:
        if model == 'DeepDTA':
            return title_set, abstract_set, y_set
        else:
            return x_set, y_set
    else:
        if model == 'DeepDTA':
            x_train_title, x_train_abstract, y_train, x_val_title, x_val_abstract, y_val = train_val_split(
                x_train=title_set, y_train=y_set, validation_percentage=validation_percentage, seed_value=seed_value,
                abstract_set=abstract_set, model='DeepDTA')
            return x_train_title, x_train_abstract, y_train, x_val_title, x_val_abstract, y_val
        else:
            x_train, y_train, x_val, y_val = train_val_split(x_set, y_set, validation_percentage=validation_percentage,
                                                             seed_value=seed_value)
            print('Training set with {} samples'.format(x_train.shape[0]))
            print('Validation set with {} samples'.format(x_val.shape[0]))
            return x_train, y_train, x_val, y_val


def Bert_preprocessing(x_set, y_set=None, config=None, nmr_sentences=1, validation_percentage=None, seed_value=None):
    padding = config.padding
    truncating = config.truncating

    tokenized_docs = []
    for doc in x_set['Document']:
        if nmr_sentences == 1:
            marked_text = "[CLS] " + doc.fulltext_string + " [SEP]"
        elif nmr_sentences == 2:
            marked_text = "[CLS] " + doc.title_string + " [SEP]" + doc.abstract_string + " [SEP]"
        else:
            raise ValueError('Bert can only take 1 or 2 sentences')

        tokenized_docs.append(config.tokenizer.tokenize(marked_text))

    indexed_docs = []
    for doc in tokenized_docs:
        indexed_docs.append(config.tokenizer.convert_tokens_to_ids(doc))

    max_sent_len = config.max_sent_len
    indexed_docs = pad_sequences(indexed_docs, maxlen=max_sent_len, padding=padding, truncating=truncating)

    segment_ids = []
    masks = []
    for id_doc in range(len(indexed_docs)):
        if nmr_sentences == 1:
            segment_ids.append([0] * len(indexed_docs[0]))
        elif nmr_sentences == 2:
            ids = [1] * len(indexed_docs[0])
            sep_id = config.tokenizer.convert_tokens_to_ids("[SEP]")
            sep_pos = np.where(indexed_docs[id_doc] == sep_id)[0][0]
            ids[:sep_pos + 1] = [0] * int(sep_pos + 1)
            segment_ids.append(ids)
        mask = []
        for index in indexed_docs[id_doc]:
            if index != 0:
                mask.append(1)
            else:
                mask.append(0)
        masks.append(mask)

    x_set = [np.asarray(indexed_docs, dtype='int32'),
             np.asarray(masks, dtype='int32'),
             np.asarray(segment_ids, dtype='int32')]

    if y_set is not None:
        y_set = np.asarray(y_set)

    if not validation_percentage or validation_percentage == 0:
        if y_set is not None:
            return x_set, y_set
        else:
            return x_set
    else:
        x_train, y_train, x_val, y_val = train_val_split(x_set, y_set, validation_percentage=validation_percentage,
                                                         seed_value=seed_value, model='bert')
        print('Training set with {} samples'.format(x_train[0].shape[0]))
        print('Validation set with {} samples'.format(x_val[0].shape[0]))
        return x_train, y_train, x_val, y_val


def average_precision(y_test_df, yhat_probs):
    yhat_probs_copy = yhat_probs.copy()
    pos_y_test = y_test_df[y_test_df == 1]
    gold_positive_ids = pos_y_test.index
    yhat_probs_df = pd.Series(data=yhat_probs_copy, index=y_test_df.index)
    yhat_probs_df[yhat_probs_df < 0.50] = - (1 - yhat_probs_df)
    sorted_yhat = yhat_probs_df.sort_values(ascending=False)
    avg_prec = prediction_count = correct = 0
    for id, confidence_score in sorted_yhat.iteritems():
        prediction_count += 1
        if id in gold_positive_ids:
            correct += 1
            avg_prec += correct / prediction_count
    avg_prec /= len(gold_positive_ids)
    return avg_prec


def plot_training_history(history_dict, config=None, path=None):
    acc = history_dict.history['accuracy']
    val_acc = history_dict.history['val_accuracy']
    loss = history_dict.history['loss']
    val_loss = history_dict.history['val_loss']
    # f1_history = history_dict.history['f1_score']
    # val_f1_history = history_dict.history['val_f1_score']

    epochs_plot = range(1, len(acc) + 1)

    palette = sns.color_palette('colorblind', 20)

    plt.rcParams['figure.figsize'] = [13, 5]

    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.plot(epochs_plot, acc, label='Training acc', color=palette[0])
    ax1.plot(epochs_plot, val_acc, label='Validation acc', color=palette[1])

    ax1.set_title('Accuracy')
    ax1.set_xlabel('epochs')
    ax1.legend()

    ax2.plot(epochs_plot, loss, label='Training loss', color=palette[0])
    ax2.plot(epochs_plot, val_loss, label='Validation loss', color=palette[1])
    ax2.set_title('Loss')
    ax2.set_xlabel('epochs')
    ax2.legend()

    # ax3.plot(epochs_plot, f1_history, label='Training f1', color = palette[0])
    # ax3.plot(epochs_plot, val_f1_history, label='Validation f1', color = palette[1])

    # ax3.set_title('F1 score')
    # ax3.set_xlabel('epochs')
    # f.tight_layout()
    # ax3.legend()

    if config:
        f.savefig(config.model_id_path / (config.model_id + '_train'))
    elif path:
        f.savefig(path)


def plot_roc_n_pr_curves(y_test, yhat_probs, config=None, path=None):
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

    ax1.plot([0, 1], [0, 1], linestyle='--', label='Random')
    fpr, tpr, _ = roc_curve(y_test, yhat_probs)
    roc_auc = roc_auc_score(y_test, yhat_probs)
    ax1.plot(fpr, tpr, marker='.', label='Trained Model')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.legend()
    ax1.set_title('ROC Curve')

    no_skill = len(y_test[y_test == 1]) / len(y_test)
    ax2.plot([0, 1], [no_skill, no_skill], linestyle='--', label='"No Skill"')
    precision, recall, _ = precision_recall_curve(y_test, yhat_probs)
    ax2.plot(recall, precision, marker='.', label='Trained Model')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.legend()
    ax2.set_title('Precision-Recall Curve')
    pr_auc = auc(recall, precision)

    if config:
        f.savefig(config.model_id_path / (config.model_id + '_test'))
    elif path:
        f.savefig(path)

    return roc_auc, pr_auc


def one_hot_words(x_train, num_words):
    data = []
    for doc in x_train["Document"]:
        data.append(doc.fulltext_string)

    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(samples)

    sequences = tokenizer.texts_to_sequences(samples)

    one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))


def one_hot_characters(x_train, max_sent_len):
    data = []
    for doc in x_train["Document"]:
        data.append(doc.fulltext_string)

    characters = string.printable  # All printable ASCII characters.
    token_index = dict(zip(characters, range(1, len(characters) + 1)))

    results = np.zeros((len(samples), max_sent_length, max(token_index.values()) + 1))
    for i, sample in enumerate(samples):
        for j, character in enumerate(sample[:max_sent_length]):
            index = token_index.get(character)
            results[i, j, index] = 1.
    print(results)