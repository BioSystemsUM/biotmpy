import pickle
from pathlib import Path
import os
import pandas as pd
from openpyxl import load_workbook
from tensorflow.keras.models import load_model



class DLConfig:
    
    def __init__(self, path=None, model_name=None, tokenizer=None,max_nb_words=None, max_sent_len=None, 
                    max_nb_sentences=None, embeddings=None, embedding_dim=None, embedding_path=None, embedding_format = None,
                    stop_words=None, embedding_matrix=None, padding=None, truncating=None, oov_token=None, lower=None,
                    remove_punctuation=None, split_by_hyphen=None, lemmatization=None, stems=None, nmr_sentences=None,
                    seed_value=None, epochs=None, batch_size=None, learning_rate=None, validation_percentage=None,
                    patience = None, keras_callbacks=False, model_id = None):

        self.model_name = model_name
        if self.model_name:
            self.create_model_folder()
        self.model_id=model_id
        self.model_id_path = None
        if self.model_id is None and self.model_name:
            self.set_model_id()
        self.path = path
        self.tokenizer=tokenizer
        self.max_sent_len = max_sent_len
        self.max_nb_words = max_nb_words
        self.max_nb_sentences = max_nb_sentences
        self.nmr_sentences = nmr_sentences
        self.embeddings = embeddings
        self.embedding_dim = embedding_dim
        self.embedding_path = embedding_path
        self.embedding_format = embedding_format
        self.stop_words = stop_words
        self.embedding_matrix = embedding_matrix
        self.padding = padding
        self.truncating = truncating
        self.oov_token = oov_token
        self.lower = lower
        self.remove_punctuation = remove_punctuation
        self.split_by_hyphen = split_by_hyphen
        self.lemmatization = lemmatization
        self.stems = stems
        self.seed_value = seed_value
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.validation_percentage = validation_percentage
        self.patience = patience
        self.keras_callbacks = keras_callbacks
        self.train_acc = None
        self.train_f1_score = None
        self.test_avg_prec = None
        self.test_acc = None
        self.test_prec = None
        self.test_recall = None
        self.test_f1_score = None
        self.test_roc_auc = None
        self.test_pr_auc = None
        self.test_kappa = None
        self.test_mcc = None
        self.test_true_neg = None
        self.test_false_pos = None
        self.test_false_neg = None
        self.test_true_pos = None



    def create_model_folder(self):
        if not os.path.exists('models'):
            os.mkdir('models')
        directory = Path('models/' + self.model_name)
        if not os.path.exists(directory):
            os.mkdir(directory)

    def set_model_id(self):
        output_path = Path('models/' + self.model_name + '/' + 'results_'+ self.model_name + '.xlsx')
        if os.path.exists(output_path):
            reader = pd.read_excel(output_path)
            self.model_id = self.model_name + '_' + str(len(reader))
        else:
            self.model_id = self.model_name + '_0'

        self.model_id_path = Path('models/' + self.model_name + '/' + self.model_id)
        if not os.path.exists(self.model_id_path):
            os.mkdir(self.model_id_path)

    def save(self, path=None):
        if path:
            with open(path, 'wb') as config_path:
                pickle.dump(self, config_path)
        else:
            self.path = self.model_id_path / 'config.txt'
            with open(self.path, 'wb') as config_path:
                pickle.dump(self, config_path)

    def load(self, path):
        with open(path, 'rb') as config_path:
            return pickle.load(config_path)

    def write_report(self):
        attrib_dict = self.__dict__.items()
        to_remove = ['model_name', 'model_id_path', 'path', 'tokenizer', 'embedding_path', 'embedding_matrix']
        data = {}
        for tup in attrib_dict:
            if tup[0] not in to_remove:
                if tup[0] == 'stop_words':
                    if tup[1] is None:
                        data[tup[0]] = ['No']
                    else:
                        data[tup[0]] = ['Removed']
                else:
                    if tup[1] == False:
                        data[tup[0]] = ['False']
                    elif tup[1] == True:
                        data[tup[0]] = ['True']
                    else:
                        data[tup[0]] = [tup[1]]
        print(data)
        df = pd.DataFrame(data)

        if not os.path.exists('../pipelines/models/' + self.model_name):
            os.mkdir('../pipelines/models/' + self.model_name)
        excel_path = Path('../pipelines/models/' + self.model_name + '/' + 'results_' + self.model_name + '.xlsx')

        if not os.path.exists(excel_path):
            writer = pd.ExcelWriter(excel_path, engine='xlsxwriter')
            df.to_excel(writer, sheet_name=self.model_name, index=False)
            writer.save()
        else:
            writer = pd.ExcelWriter(excel_path, engine='openpyxl')
            writer.book = load_workbook(excel_path)
            writer.sheets = dict((ws.title, ws) for ws in writer.book.worksheets)
            reader = pd.read_excel(excel_path)
            df.to_excel(writer, index=False, header=False, startrow=len(reader) + 1, sheet_name=self.model_name)
            writer.close()

        with open(self.model_id_path / "config_description.txt", "w") as config_txt:
            config_txt.write(str(data))


    def load_tensorflow_model(self, path):
        model = load_model(path)
        return model

    def save_data(self, x_train, y_train, x_val=None, y_val=None, path=None):
        data = {'x_train': x_train, 'y_train':y_train,'x_val':x_val, 'y_val':y_val}
        if path:
            with open(path, 'wb') as data_path:
                pickle.dump(data, data_path)
        else:
            self.path = self.model_id_path / 'config.txt'
            with open(self.path, 'wb') as data_path:
                pickle.dump(data, data_path)

    def load_data(self, path):
        with open(path, 'rb') as data_path:
            return pickle.load(data_path)



