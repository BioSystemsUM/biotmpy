model_name= 'hp_han'



import sys 
sys.path.append('../')
import os
import tensorflow 
import numpy as np
import random

if not os.path.isdir('hp_results/'):
    os.mkdir('hp_results')

global seed_value
seed_value = 123123
#seed_value = None

environment_name = sys.executable.split('/')[-3]
print('Environment:', environment_name)
os.environ[environment_name] = str(seed_value)

np.random.seed(seed_value)
random.seed(seed_value)
tensorflow.random.set_seed(seed_value)

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import tensorflow.compat.v1.keras.backend as K
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
K.set_session(session)

multiple_gpus = [0,1,2,3]
#multiple_gpus = None

import os
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


if multiple_gpus:
    devices = []
    for gpu in multiple_gpus:
        devices.append('/gpu:' + str(gpu))    
    strategy = tensorflow.distribute.MirroredStrategy(devices=devices)

else:
    # Get the GPU device name.
    device_name = tensorflow.test.gpu_device_name()
    # The device name should look like the following:
    if device_name == '/device:GPU:0':
        print('Using GPU: {}'.format(device_name))
    else:
        raise SystemError('GPU device not found')

    os.environ["CUDA_VISIBLE_DEVICES"] = device_name
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

from src.biotmpy.wrappers.bioc_wrapper import bioc_to_docs, bioc_to_relevances
from src.biotmpy.wrappers.pandas_wrapper import relevances_to_pandas, docs_to_pandasdocs
from src.biotmpy.mlearning import DL_preprocessing
from src.biotmpy.mlearning.dl_models import Hierarchical_Attention_GRU, Hierarchical_Attention_LSTM,Hierarchical_Attention_LSTM2, Hierarchical_Attention_LSTM3
from src.biotmpy.mlearning.dl_models import Hierarchical_Attention_Context
from src.biotmpy.mlearning.dl_models import DeepDTA
from src.biotmpy.mlearning import compute_embedding_matrix, glove_embeddings_2
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score, auc, roc_curve, precision_recall_curve
from sklearn.metrics import confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from src.biotmpy.mlearning import plot_training_history
from src.biotmpy.mlearning import DLConfig
from src.biotmpy.mlearning import average_precision
from tensorflow.keras.preprocessing import text
from src.biotmpy.mlearning import plot_roc_n_pr_curves
from sklearn.model_selection import cross_val_score
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, LSTM, RNN, Bidirectional, Flatten, Activation,     RepeatVector, Permute, Multiply, Lambda, Concatenate, BatchNormalization
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import SGD, Adam
from sklearn.metrics import confusion_matrix
from src.biotmpy.mlearning import AttentionLayer
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import TimeDistributed, GRU
import tensorflow.keras.backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import RMSprop, Adagrad, Adam, SGD
from transformers import TFBertModel
from src.biotmpy.mlearning.attention_context import AttentionWithContext
from tensorflow.keras.layers import SpatialDropout1D
from transformers import TFBertForSequenceClassification, BertConfig
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import f1_score, recall_score, precision_score
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
from nltk.corpus import stopwords
import seaborn as sns
import pandas as pd
import os
import pickle
from kerastuner import Hyperband
import pickle
import json


train_dataset_path = '../data/PMtask_Triage_TrainingSet.xml'
test_dataset_path = '../data/PMtask_Triage_TestSet.xml'



dl_config = DLConfig(model_name=model_name, seed_value=seed_value)
dl_config.stop_words = set(stopwords.words('english'))           
#dl_config.stop_words = None
dl_config.lower = True               
dl_config.remove_punctuation = False
dl_config.split_by_hyphen = True
dl_config.lemmatization = False           
dl_config.stems = False                      


docs_train = bioc_to_docs(train_dataset_path, dl_config=dl_config)
relevances_train = bioc_to_relevances(train_dataset_path, 'protein-protein')


x_train_df = docs_to_pandasdocs(docs_train)
y_train_df = relevances_to_pandas(x_train_df, relevances_train)

#Parameters
dl_config.padding = 'post'            #'pre' -> default; 'post' -> alternative
dl_config.truncating = 'post'         #'pre' -> default; 'post' -> alternative      #####
dl_config.oov_token = 'OOV'

dl_config.max_sent_len = 50      #sentences will have a maximum of "max_sent_len" words    #400/500
dl_config.max_nb_words = 100_000      #it will only be considered the top "max_nb_words" words in the dataset
dl_config.max_nb_sentences = 15    # set only for the hierarchical attention model!!!



dl_config.embeddings = 'biowordvec'
dl_config.embedding_path = './embeddings/12551780'
dl_config.embedding_dim = 200
dl_config.embedding_format = 'word2vec'



dl_config.tokenizer = text.Tokenizer(num_words=dl_config.max_nb_words, oov_token=dl_config.oov_token)

x_train, y_train, x_val, y_val = DL_preprocessing(x_train_df, y_train_df,
    dl_config=dl_config, dataset='train',
    validation_percentage=10,
    seed_value=dl_config.seed_value)



dl_config.embedding_matrix = compute_embedding_matrix(dl_config, embeddings_format = dl_config.embedding_format)



global max_sent_len, max_nb_sentences, vocab_size, embed_dim, embedding_matrix
max_sent_len = dl_config.max_sent_len
max_nb_sentences = dl_config.max_nb_sentences
vocab_size = dl_config.embedding_matrix.shape[0]
embed_dim = dl_config.embedding_matrix.shape[1]
embedding_matrix = dl_config.embedding_matrix


def Han_hyper(hp):
    embedding_layer = Embedding(vocab_size, embed_dim, weights=[embedding_matrix],
                                input_length=max_sent_len, trainable=False, name='word_embedding')
    
    
    word_input = Input(shape=(max_sent_len,), dtype='int32')
    word = embedding_layer(word_input)
    
    word = SpatialDropout1D(rate=hp.Float('dropout_1', min_value=0.0, max_value=0.5, default=0.2, step=0.1), seed=seed_value)(word)
    
    
    layer_1 = hp.Choice('layer_1',['lstm', 'gru'])
    layer_1_units = hp.Choice('layer1_units', values=[64,128,256,512], default=128)

    if layer_1 == 'lstm':
        word = Bidirectional(LSTM(layer_1_units, return_sequences=True))(word)
    elif layer_1 == 'gru':
        word = Bidirectional(GRU(layer_1_units, return_sequences=True))(word)

    word_out = AttentionWithContext()(word)
    wordEncoder = Model(word_input, word_out)

    
    sente_input = Input(shape=(max_nb_sentences, max_sent_len), dtype='int32')
    sente = TimeDistributed(wordEncoder)(sente_input)
    sente = SpatialDropout1D(rate=hp.Float('dropout_2', min_value=0.0, max_value=0.5, default=0.2, step=0.1),
                             seed = seed_value)(sente)
    
    
    layer_2 = hp.Choice('layer_2', values=['lstm', 'gru'])
    layer_2_units = hp.Choice('layer2_units', values=[64,128,256,512], default=128)
    if layer_2 == 'lstm':
        sente = Bidirectional(LSTM(layer_2_units, return_sequences=True))(sente)
    elif layer_2 == 'gru':
        sente = Bidirectional(GRU(layer_2_units, return_sequences=True))(sente)
    sente = AttentionWithContext()(sente)
    preds = Dense(1, activation='sigmoid')(sente)
    model = Model(sente_input, preds)
    
    optimizer_value = hp.Choice('optimizer_name', values=[ 'adagrad', 'rmsprop','adam', 'sgd'])
    learning_rate = hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])

    if optimizer_value=='adagrad':
        optimizer=Adagrad(lr=learning_rate)
    elif optimizer_value=='adam':
        optimizer=Adam(lr=learning_rate)
    elif optimizer_value=='rmsprop':
        optimizer=RMSprop(lr=learning_rate)
    elif optimizer_value=='sgd':
        optimizer=SGD(lr=learning_rate)

        

    model.compile(loss='binary_crossentropy',
                optimizer=optimizer,
                metrics=['accuracy'])

    print(model.summary())
    return model


tuner = Hyperband(
    Han_hyper,
    max_epochs=40,
    objective='val_accuracy',
    seed = seed_value,
    directory='./hp_results/' + model_name,
    hyperband_iterations=2,
    overwrite=True)

tuner.search(x_train,y_train, epochs=40, validation_data=(x_val, y_val))

result = tuner.get_best_hyperparameters(1)[0].values


with open('hp_results/' + model_name + '/best_hyperparameters.txt', 'w') as file:
     json.dump(result, file, indent = 4);



with open('./hp_results/' + model_name + 'tuner_han.pkl', 'wb') as f:
    pickle.dump(tuner, f)




