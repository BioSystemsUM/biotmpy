# gpu_id = None
# if len(sys.argv) == 2:
#    gpu_id = sys.argv[1]
# if not gpu_id:
#    raise Exception('insert gpu_id')

# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#
# os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

import sys
sys.path.append('../')
from wrappers.bioc_wrapper import bioc_to_docs, bioc_to_relevances
from wrappers.pandas_wrapper import relevances_to_pandas, docs_to_pandasdocs
from mlearning.dl import DL_preprocessing
from mlearning.dl_models import Burns_CNN, Burns_LSTM, Chollet_DNN, DNN, Burns_CNN2, Burns_BiLSTM, Burns_CNN3, Burns_CNNBiLSTM
from mlearning.dl_models import Hierarchical_Attention, Hierarchical_Attention_v2, DeepDTA
from mlearning.embeddings import compute_embedding_matrix, glove_embeddings_2
from mlearning.ml import serialize_config, deserialize_config
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from mlearning.dl_config import DLConfig
from tensorflow.keras.preprocessing import text
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
import seaborn as sns
from openpyxl import load_workbook
import pandas as pd
import os
from keras import backend as K
import tensorflow as tf
from tfdeterminism import patch
patch()


# gpu_id = None
# if len(sys.argv) == 2:
#    gpu_id = sys.argv[1]
# if not gpu_id:
#    raise Exception('insert gpu_id')
# gpus  = tf.config.experimental.list_physical_devices('GPU')

seed_value= 123123

os.environ['pythongpu'] = str(seed_value)

tf.random.set_seed(seed_value)

# with K.tf.device(gpu):
#     config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=4,\
#            inter_op_parallelism_threads=4, allow_soft_placement=True,\
#            device_count = {'CPU' : 1, 'GPU' : 1})
#     config.gpu_options.allow_growth = True
#     session = tf.compat.v1.Session(config=config)
#     K.set_session(session)
from keras import backend as K
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.compat.v1.Session(config=config)
K.set_session(sess)


#Parameters
model_name= 'deepdta'

stop_words = set(stopwords.words('english'))            #####
#stop_words = None
lower = True                #####
remove_punctuation = False
split_by_hyphen = True
lemmatization = False           #####
stems = False                       #####
padding = 'pre'            #'pre' -> default; 'post' -> alternative
truncating = 'pre'         #'pre' -> default; 'post' -> alternative      #####
oov_token = 'OOV'

epochs = 20
batch_size = 64     # e aumentar o batch
learning_rate = 0.0001   #experimentar diminuir

max_sent_len = 400      #sentences will have a maximum of "max_sent_len" words    #400/500
max_nb_words = 100_000      #it will only be considered the top "max_nb_words" words in the dataset
max_nb_sentences = None    # set only for the hierarchical attention model!!!

embeddings = 'pubmed_pmc'

validation_percentage = 10


if embeddings == 'glove':
    embedding_path = 'D:/desktop/tese/embeddings/glove/glove.6B.300d.txt'
    #embedding_path = '/home/malves/embeddings/glove/glove.840B.300d.txt'
    embedding_dim = 300
    embedding_format = 'glove'

elif embeddings == 'biowordvec':
    #BioWordVec_extrinsic   #lowercase #200dimensions
    embedding_path = 'D:/desktop/tese/embeddings/biowordvec/bio_embedding_extrinsic'
    #embedding_path = '/home/malves/embeddings/biowordvec/bio_embedding_extrinsic'
    embedding_dim = 200
    embedding_format = 'word2vec'

elif embeddings == 'pubmed_pmc':   #200 dimensions
    embedding_path = 'D:/desktop/tese/embeddings/pubmed_pmc/PubMed-and-PMC-w2v.bin'
    #embedding_path = '/home/malves/embeddings/pubmed_pmc/PubMed-and-PMC-w2v.bin'
    embedding_dim = 200
    embedding_format = 'word2vec'

else:
    raise Exception("Please Insert Embeddings Type")


es_patience = 30    #early-stopping patience
# keras_callbacks = [
#         EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=es_patience),
#         ModelCheckpoint('best_model.h5', monitor='val_f1_score', mode='max', verbose=1, save_best_only=True)
# ]

keras_callbacks = None


train_dataset_path = '../datasets/PMtask_Triage_TrainingSet.xml'
test_dataset_path = '../datasets/PMtask_Triage_TestSet.xml'
output_excel = 'metrics/results_' + model_name + '.xlsx'

#Pipeline
#Load Data
docs_train = bioc_to_docs(train_dataset_path, stop_words=stop_words, lower=lower, remove_punctuation=remove_punctuation,
                            split_by_hyphen=split_by_hyphen, lemmatization=lemmatization, stems=stems)
relevances_train = bioc_to_relevances(train_dataset_path, 'protein-protein')


x_train = docs_to_pandasdocs(docs_train)
y_train = relevances_to_pandas(x_train, relevances_train)


#Preprocessing for Training Data
path = '../models/configs/' + model_name + '.txt'

tokenizer = text.Tokenizer(num_words=max_nb_words, oov_token=oov_token)

our_sent = tokenizer.texts_to_sequences([x_train['Document'][0].fulltext_string])
for i, tok in enumerate(x_train['Document'][0].fulltext_tokens):
    print(i, ': ', tok)

dl_config = DLConfig(path=path, tokenizer=tokenizer, max_sent_len=max_sent_len, max_nb_sentences=max_nb_sentences,
                    embedding_dim=embedding_dim, embedding_path=embedding_path, max_nb_words=max_nb_words)

# x_train, y_train, x_val, y_val = DL_preprocessing(x_train, y_train, dl_config, set='train',
#                                     validation_percentage = validation_percentage, seed_value=seed_value,
#                                     padding=padding, truncating=truncating)

x_train_title, x_train_abstract, y_train, x_val_title, x_val_abstract, y_val = DL_preprocessing(x_train, y_train, dl_config, set='train',
                                    validation_percentage = validation_percentage, seed_value=seed_value,
                                    padding=padding, truncating=truncating, model='DeepDTA')

embedding_matrix = compute_embedding_matrix(dl_config, embeddings_format = embedding_format)



#Deep Learning models
#model = Hierarchical_Attention(embedding_matrix, dl_config, seed_value=seed_value)
#model = Hierarchical_Attention_v2(embedding_matrix, dl_config, seed_value=seed_value)
#model = Burns_CNNBiLSTM(embedding_matrix, dl_config, seed_value=seed_value)
#model = Burns_CNN(embedding_matrix, dl_config, seed_value=seed_value)
model = DeepDTA(embedding_matrix, dl_config)



# history = model.fit(x_train, y_train,
#                     epochs=epochs,
#                     batch_size=batch_size,
#                     validation_data=(x_val, y_val),
#                     callbacks=keras_callbacks)

history = model.fit(([x_train_title, x_train_abstract]), y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=([x_val_title, x_val_abstract], y_val),
                    callbacks=keras_callbacks)

dl_config.classifier = model

#clf = load_model('best_model.h5')


#EVALUATION
#_, train_acc, train_f1_score = model.evaluate(x_train, y_train, verbose=0)
_, train_acc, train_f1_score = model.evaluate([x_train_title, x_train_abstract], y_train, verbose=0)



print('Training Accuracy: %.3f' % (train_acc))
print('Training F1_score: %.3f' % (train_f1_score))

print(history.history)


if os.path.exists(output_excel):
    reader = pd.read_excel('metrics/results_'+model_name+'.xlsx')
    model_id = model_name + '_' + str(len(reader))
else:
    model_id = model_name + '_0'


model.save_weights('models/weights/' + model_id + '.h5')


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
f1_history = history.history['f1_score']
val_f1_history = history.history['val_f1_score']


epochs_plot = range(1, len(acc) + 1)

palette = sns.color_palette('colorblind', 20)

plt.plot(epochs_plot, acc, label='Training acc', color = palette[0])
plt.plot(epochs_plot, val_acc, label='Validation acc', color = palette[1])
plt.plot(epochs_plot, loss, label='Training loss', color = palette[2])
plt.plot(epochs_plot, val_loss, label='Validation loss', color = palette[7])
plt.title('Learning Curves')
plt.xlabel('epochs')
plt.legend()
plt.clf()
plt.savefig('metrics/plots/%s.png' % model_id)

plt.plot(epochs_plot, f1_history, label='Training f1', color = palette[3])
plt.plot(epochs_plot, val_f1_history, label='Validation f1', color = palette[6])

plt.title('Learning Curves')
plt.xlabel('epochs')
plt.legend()
plt.draw()
plt.savefig('metrics/plots/f1_score/%s.png' % model_id)


#### TEST SET
#Preprocessing for Test Data
docs_test = bioc_to_docs(test_dataset_path, stop_words=stop_words, lower=lower, remove_punctuation=remove_punctuation,
                        split_by_hyphen=split_by_hyphen, lemmatization=lemmatization, stems=stems)
relevances_test = bioc_to_relevances(test_dataset_path, 'protein-protein')

x_test = docs_to_pandasdocs(docs_test)
y_test = relevances_to_pandas(x_test, relevances_test)

#x_test, y_test = DL_preprocessing(x_test, y_test, dl_config, set = 'test', padding=padding, truncating=truncating)

x_test_title, x_test_abstract, y_test = DL_preprocessing(x_test, y_test, dl_config, set = 'test', padding=padding, truncating=truncating, model='DeepDTA')


model.load_weights('models/weights/' + model_id + '.h5')


#Deep Learning predictions
#yhat_probs = model.predict(x_test, verbose=0)
yhat_probs = model.predict([x_test_title, x_test_abstract], verbose=0)
yhat_probs = yhat_probs[:, 0]

yhat_classes = np.where(yhat_probs > 0.5, 1, yhat_probs)
yhat_classes = np.where(yhat_classes < 0.5, 0, yhat_classes).astype(np.int64)


# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_test, yhat_classes)
print('Accuracy: %f' % accuracy)

# precision tp / (tp + fp)
precision = precision_score(y_test, yhat_classes)
print('Precision: %f' % precision)

# recall: tp / (tp + fn)
recall = recall_score(y_test, yhat_classes)
print('Recall: %f' % recall)

# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_test, yhat_classes)
print('F1 score: %f' % f1)

# kappa
kappa = cohen_kappa_score(y_test, yhat_classes)
print('Cohens kappa: %f' % kappa)
# ROC AUC
auc = roc_auc_score(y_test, yhat_probs)
print('ROC AUC: %f' % auc)

# confusion matrix
matrix = confusion_matrix(y_test, yhat_classes)
print(matrix)


tn, fp, fn, tp = confusion_matrix(y_test, yhat_classes).ravel()








#save parameters
d = {'Model':[], 'Emb':[], 'Emb_Dim':[], 'Learning_rate': [], 'train_acc':[], 'train_f1':[],'test_acc':[], 'test_prec':[], 'test_recall':[], 'test_f1':[],
     'roc_auc':[], 'TP':[], 'TN':[], 'FP':[], 'FN':[],'epochs':[], 'batch_size':[], 'stop_words':[],'lower':[],'remove_punc':[],
     'max_sent_len':[], 'max_nb_words':[], 'max_nb_sentences':[], 'oov_token':[], 'val_percentage':[], 'split_by_hypen':[], 'seed_value':[],
     'history_acc':[], 'history_loss':[],'history_f1':[],'history_val_acc':[], 'history_val_loss':[],'history_val_f1':[], 'lemmatization':[],
     'truncating':[], 'padding':[], 'stems':[]}
df = pd.DataFrame(d)

if not os.path.exists(output_excel):
    writer = pd.ExcelWriter(output_excel, engine='xlsxwriter')
    df.to_excel(writer, sheet_name=model_name, index=False)
    writer.save()


if stems:
    d['stems'].append('yes')
else:
    d['stems'].append('no')
d['test_acc'].append(accuracy)
d['test_prec'].append(precision)
d['test_recall'].append(recall)
d['test_f1'].append(f1)
d['roc_auc'].append(auc)
d['TP'].append(tp)
d['TN'].append(tn)
d['FP'].append(fp)
d['FN'].append(fn)
d['history_acc'].append(acc[-1])
d['history_loss'].append(loss[-1])
d['history_f1'].append(f1_history[-1])
d['history_val_acc'].append(val_acc[-1])
d['history_val_loss'].append(val_loss[-1])
d['history_val_f1'].append(val_f1_history[-1])
d['truncating'].append(truncating)
d['padding'].append(padding)
d['epochs'].append(epochs)
d['train_acc'].append(train_acc)
d['train_f1'].append(train_f1_score)
d['batch_size'].append(batch_size)
d['max_sent_len'].append(max_sent_len)
d['max_nb_words'].append(max_nb_words)
d['max_nb_sentences'].append((max_nb_sentences))
d['seed_value'].append(seed_value)
d['Learning_rate'].append(learning_rate)
d['Model'].append(model_id)
d['val_percentage'].append(validation_percentage)
if stop_words:
    d['stop_words'].append('yes')
else:
    d['stop_words'].append('no')
if lower:
    d['lower'].append('yes')
else:
    d['lower'].append('no')
if oov_token:
    d['oov_token'].append('yes')
else:
    d['oov_token'].append('no')
if remove_punctuation:
    d['remove_punc'].append('yes')
else:
    d['remove_punc'].append('no')
if split_by_hyphen:
    d['split_by_hypen'].append('yes')
else:
    d['split_by_hypen'].append('no')
if lemmatization:
    d['lemmatization'].append('yes')
else:
    d['lemmatization'].append('no')

d['Emb'].append(embeddings)
d['Emb_Dim'].append(embedding_dim)

print(d)
df = pd.DataFrame(d)
writer = pd.ExcelWriter(output_excel, engine='openpyxl')
# try to open an existing workbook
writer.book = load_workbook(output_excel)
# copy existing sheets
writer.sheets = dict((ws.title, ws) for ws in writer.book.worksheets)
# read existing file
reader = pd.read_excel('metrics/results_'+ model_name + '.xlsx')
# write out the new sheet
df.to_excel(writer,index=False,header=False,startrow=len(reader)+1, sheet_name=model_name)

writer.close()





