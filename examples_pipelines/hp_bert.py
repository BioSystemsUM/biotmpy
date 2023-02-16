model_name= 'hp_bert'



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
from src.biotmpy.mlearning import Bert_preprocessing
from transformers import BertTokenizer
from transformers import AutoTokenizer
import tensorflow_addons as tfa
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
#dl_config.stop_words = set(stopwords.words('english'))           
dl_config.stop_words = None
dl_config.lower = False               
dl_config.remove_punctuation = False
dl_config.split_by_hyphen = False
dl_config.lemmatization = False           
dl_config.stems = False                      


docs_train = bioc_to_docs(train_dataset_path, dl_config=dl_config)
relevances_train = bioc_to_relevances(train_dataset_path, 'protein-protein')


x_train_df = docs_to_pandasdocs(docs_train)
y_train_df = relevances_to_pandas(x_train_df, relevances_train)

#Parameters
dl_config.padding = 'post'            #'pre' -> default; 'post' -> alternative
dl_config.truncating = 'post'         #'pre' -> default; 'post' -> alternative      #####

dl_config.max_sent_len = 512      #sentences will have a maximum of "max_sent_len" words
dl_config.nmr_sentences = 1      #[1 or 2]

dl_config.validation_percentage = 10

dl_config.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

x_train, y_train, x_val, y_val =  Bert_preprocessing(x_train_df, y_train_df, 
                                                     dl_config, 
                                                     nmr_sentences = dl_config.nmr_sentences, 
                                                     validation_percentage = dl_config.validation_percentage, 
                                                     seed_value=dl_config.seed_value)



global max_sent_len
max_sent_len = dl_config.max_sent_len


def Bert_hyper(hp):
    idx = Input((max_sent_len), dtype="int32", name="input_idx")
    masks = Input((max_sent_len), dtype="int32", name="input_masks")
    segments = Input((max_sent_len), dtype="int32", name="input_segments")
    

    bert_model = TFBertModel.from_pretrained("bert-base-uncased")
    embedding = bert_model([idx, masks, segments])[0]

    ## fine-tuning
    x = Bidirectional(LSTM(hp.Choice('lstm_units', values=[64,128,256], default=128), return_sequences=True))(embedding)
    x = Dropout(rate=hp.Float('dropout_1', min_value=0.0, max_value=0.5, default=0.2, step=0.1), seed=seed_value)(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(hp.Choice('dense_units', values=[64,128,256], default=64), activation="relu")(x)
    x = Dropout(rate=hp.Float('dropout_2', min_value=0.0, max_value=0.5, default=0.2, step=0.1), seed=seed_value)(x)

    y_out = Dense(1, activation='sigmoid')(x)


    model = Model([idx, masks, segments], y_out)

    for layer in model.layers[:4]:
        layer.trainable = False


    
    optimizer_value = hp.Choice('optimizer_name', values=['adam', 'adamw'])
    learning_rate = hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])

    if optimizer_value=='adam':
        optimizer=Adam(lr=learning_rate)
    elif optimizer_value=='adamw':
        optimizer=tfa.optimizers.AdamW(weight_decay = 0.01, learning_rate=learning_rate)


    model.compile(loss='binary_crossentropy',
                optimizer=optimizer,
                metrics=['accuracy'])

    print(model.summary())
    return model


tuner = Hyperband(
    Bert_hyper,
    max_epochs=1,
    objective='val_accuracy',
    seed = seed_value,
    directory='./hp_results/' + model_name,
    hyperband_iterations=1,
    distribution_strategy=tf.distribute.MirroredStrategy(['/gpu:0','/gpu:1', '/gpu:2', '/gpu:3']), 	
    overwrite=True)

tuner.search(x_train,y_train, epochs=1, batch_size=32, validation_data=(x_val, y_val))

result = tuner.get_best_hyperparameters(1)[0].values



with open('hp_results/' + model_name + '/best_hyperparameters.txt', 'w') as file:
     json.dump(result, file, indent = 4);

tuner.distribution_strategy=None
tuner.hypermodel=None
with open('hp_results/' + model_name + '/tuner_bert.pkl', 'wb') as f:
    pickle.dump(tuner, f)

