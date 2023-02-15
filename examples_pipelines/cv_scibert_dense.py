model_name= 'cv_scibert_dense'

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

from src.biotmpy.wrappers.bioc_wrapper import bioc_to_docs, bioc_to_relevances
from src.biotmpy.wrappers.pandas_wrapper import relevances_to_pandas, docs_to_pandasdocs
from src.biotmpy.mlearning import DL_preprocessing
from src.biotmpy.mlearning.dl_models import Bert_Dense_opt
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
from src.biotmpy.mlearning import plot_roc_n_pr_curves, Bert_preprocessing
from transformers import BertTokenizer
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

dl_config.learning_rate = 2e-5
dl_config.epochs = 2

dl_config.batch_size=16



dl_config.k_fold=10
kfold = StratifiedKFold(n_splits=dl_config.k_fold, shuffle=True, random_state=dl_config.seed_value)

cv_avp_scores = []
cv_acc_scores=[]
cv_prec_scores = []
cv_rec_scores = []
cv_f1_scores = []
for train_index, test_index in kfold.split(x_train_df.to_numpy(), y_train_df.to_numpy()):
    print(len(train_index))
    print(len(test_index))
    K.clear_session()


    environment_name = sys.executable.split('/')[-3]
    print('Environment:', environment_name)
    os.environ[environment_name] = str(dl_config.seed_value)

    np.random.seed(dl_config.seed_value)
    random.seed(dl_config.seed_value)
    tensorflow.random.set_seed(dl_config.seed_value)

    dl_config.tokenizer = BertTokenizer.from_pretrained('./scibert_scivocab_uncased', do_lower_case=True)

    x_train, y_train = Bert_preprocessing(x_train_df.iloc[train_index,], y_train_df.iloc[train_index,],
        dl_config=dl_config, 
        validation_percentage=0,
        seed_value=dl_config.seed_value)


    scibert_path = './scibert_scivocab_uncased'

    if multiple_gpus:
        with strategy.scope():
            model = Bert_Dense_opt(dl_config, learning_rate=dl_config.learning_rate,static_bert=False, bert_name_or_path=scibert_path, bert_config=True)
    else:
        model = Bert_Dense_opt(dl_config, learning_rate=dl_config.learning_rate,static_bert=False, bert_name_or_path=scibert_path, bert_config=True)

    history = model.fit(x_train, y_train,
                        epochs=dl_config.epochs,
                        batch_size=dl_config.batch_size)

    x_test, y_test = Bert_preprocessing(x_train_df.iloc[test_index,], y_train_df.iloc[test_index,], dl_config=dl_config)

    yhat_probs = model.predict(x_test, verbose=0)
    yhat_probs = yhat_probs[:, 0]

    yhat_classes = np.where(yhat_probs > 0.5, 1, yhat_probs)
    yhat_classes = np.where(yhat_classes < 0.5, 0, yhat_classes).astype(np.int64)
    
    test_avp = average_precision(y_train_df.iloc[test_index,], yhat_probs)
    test_acc = accuracy_score(y_test, yhat_classes)
    test_prec = precision_score(y_test, yhat_classes)
    test_rec = recall_score(y_test, yhat_classes)
    test_f1 = f1_score(y_test, yhat_classes)
    cv_avp_scores.append(test_avp)
    cv_acc_scores.append(test_acc)
    cv_prec_scores.append(test_prec)
    cv_rec_scores.append(test_rec)
    cv_f1_scores.append(test_f1)


    K.clear_session()
    del model
    tf.compat.v1.reset_default_graph()

dl_config.cv_avp = cv_avp_scores
dl_config.cv_acc = cv_acc_scores
dl_config.cv_prec = cv_prec_scores
dl_config.cv_rec = cv_rec_scores
dl_config.cv_f1 = cv_f1_scores



dl_config.save()


dl_config.write_report()
