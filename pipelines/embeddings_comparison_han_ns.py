model_name= 'embeddings_comparison_han_nonsplit'

import sys 
sys.path.append('../')
import os
import tensorflow 
import numpy as np
import random


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

from wrappers.bioc_wrapper import bioc_to_docs, bioc_to_relevances
from wrappers.pandas_wrapper import relevances_to_pandas, docs_to_pandasdocs
from mlearning.dl import DL_preprocessing
from mlearning.dl_models import Hierarchical_Attention_GRU, Hierarchical_Attention_LSTM,Hierarchical_Attention_LSTM2, Hierarchical_Attention_LSTM3
from mlearning.dl_models import Hierarchical_Attention_Context
from mlearning.dl_models import DeepDTA
from mlearning.embeddings import compute_embedding_matrix, glove_embeddings_2
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
from mlearning.dl import plot_training_history
from mlearning.dl_config import DLConfig
from mlearning.dl import average_precision
from tensorflow.keras.preprocessing import text
from mlearning.dl import plot_roc_n_pr_curves
from sklearn.model_selection import cross_val_score
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
from nltk.corpus import stopwords
import seaborn as sns
import pandas as pd
import os
from keras import backend as K
import pickle

train_dataset_path = '../datasets/PMtask_Triage_TrainingSet.xml'
test_dataset_path = '../datasets/PMtask_Triage_TestSet.xml'



dl_config = DLConfig(model_name=model_name, seed_value=seed_value)
dl_config.stop_words = set(stopwords.words('english'))           
#dl_config.stop_words = None
dl_config.lower = True               
dl_config.remove_punctuation = False
dl_config.split_by_hyphen = False
dl_config.lemmatization = False           
dl_config.stems = False                      


docs_train = bioc_to_docs(train_dataset_path, dl_config=dl_config)
relevances_train = bioc_to_relevances(train_dataset_path, 'protein-protein')


x_train_df = docs_to_pandasdocs(docs_train)
y_train_df = relevances_to_pandas(x_train_df, relevances_train)

#Parameters
dl_config.padding = 'pre'            #'pre' -> default; 'post' -> alternative
dl_config.truncating = 'post'         #'pre' -> default; 'post' -> alternative      #####
dl_config.oov_token = 'OOV'

dl_config.epochs = 21
dl_config.batch_size = 32     # e aumentar o batch
dl_config.learning_rate = 0.0001   #experimentar diminuir

dl_config.max_sent_len = 50      #sentences will have a maximum of "max_sent_len" words    #400/500
dl_config.max_nb_words = 100_000      #it will only be considered the top "max_nb_words" words in the dataset
dl_config.max_nb_sentences = 15    # set only for the hierarchical attention model!!!


embeddings_list = ['glove', 'biowordvec', 'pubmed_pmc', 'pubmed_ncbi']
for embedding in embeddings_list:
    dl_config.embeddings = embedding

    if dl_config.embeddings == 'glove':
        dl_config.embedding_path = '/home/malves/embeddings/glove/glove.6B.200d.txt'
        dl_config.embedding_dim = 200
        dl_config.embedding_format = 'glove'

    elif dl_config.embeddings == 'biowordvec':
        #BioWordVec_extrinsic   #lowercase #200dimensions
        dl_config.embedding_path = '/home/malves/embeddings/biowordvec/bio_embedding_extrinsic'
        dl_config.embedding_dim = 200
        dl_config.embedding_format = 'word2vec'

    elif dl_config.embeddings == 'pubmed_pmc':   #200 dimensions
        dl_config.embedding_path = '/home/malves/embeddings/pubmed_pmc/PubMed-and-PMC-w2v.bin'
        dl_config.embedding_dim = 200
        dl_config.embedding_format = 'word2vec'

    elif dl_config.embeddings == 'pubmed_ncbi':   #100 dimensions
        dl_config.embedding_path = '/home/malves/embeddings/pubmed_ncbi/pubmed_s100w10_min.bin'
        dl_config.embedding_dim = 100
        dl_config.embedding_format = 'word2vec'

    else:
        raise Exception("Please Insert Embeddings Type")
    dl_config.k_fold=10
    kfold = StratifiedKFold(n_splits=dl_config.k_fold, shuffle=True, random_state=dl_config.seed_value)
    cv_acc_scores=[]
    for train_index, test_index in kfold.split(x_train_df.to_numpy(), y_train_df.to_numpy()):
        print(len(train_index))
        print(len(test_index))
        dl_config.tokenizer = text.Tokenizer(num_words=dl_config.max_nb_words, oov_token=dl_config.oov_token)

        x_train, y_train = DL_preprocessing(x_train_df.iloc[train_index,], y_train_df.iloc[train_index,],
            dl_config=dl_config, dataset='train',
            validation_percentage=0,
            seed_value=dl_config.seed_value)


        dl_config.embedding_matrix = compute_embedding_matrix(dl_config, embeddings_format = dl_config.embedding_format)

        if multiple_gpus:
            with strategy.scope():
                model = Hierarchical_Attention_Context(dl_config.embedding_matrix, dl_config, learning_rate=dl_config.learning_rate,
                                                       seed_value=dl_config.seed_value)
        else:
            model = Hierarchical_Attention_Context(dl_config.embedding_matrix, dl_config, learning_rate=dl_config.learning_rate,
                                                        seed_value=dl_config.seed_value)

        history = model.fit(x_train, y_train,
                            epochs=dl_config.epochs,
                            batch_size=dl_config.batch_size)

        x_test, y_test = DL_preprocessing(x_train_df.iloc[test_index,], y_train_df.iloc[test_index,], dl_config=dl_config, dataset='test')

        _, test_acc = model.evaluate(x_test, y_test, verbose=0)
        cv_acc_scores.append(test_acc)
    dl_config.cv_acc= cv_acc_scores



    dl_config.save()


    dl_config.write_report()


    #
    # model.save(dl_config.model_id_path / 'model_tf', save_format = 'tf')
    #
    #
    # model_yaml = model.to_yaml()
    # with open(dl_config.model_id_path / "model.yaml", "w") as yaml_file:
    #     yaml_file.write(model_yaml)
