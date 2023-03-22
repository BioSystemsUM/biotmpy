import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

seed_value = 123123
# seed_value = None


import sys
import numpy as np
import random
import tensorflow
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import tensorflow.compat.v1.keras.backend as K

environment_name = sys.executable.split('/')[-3]
print('Environment:', environment_name)
os.environ[environment_name] = str(seed_value)

np.random.seed(seed_value)
random.seed(seed_value)
tensorflow.random.set_seed(seed_value)
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
K.set_session(session)

from flask import Flask, render_template, request
import pandas as pd
import pubmed_reader
import sys

sys.path.append(os.path.dirname(__file__))
from biotmpy.data_structures.config import Config
from src.biotmpy.preprocessing.dl import Bert_preprocessing
from src.biotmpy.mlearning.dl_models import Bert_LSTM_opt
from src.biotmpy.wrappers.pandas_wrapper import docs_to_pandasdocs

app = Flask(__name__)

global model, config
config = Config()
config = config.load(os.path.join(os.path.dirname(__file__), 'best_model/biobert_lstm_ft_28/config.txt'))
biobert_path = os.path.join(os.path.dirname(__file__), 'best_model/biobert_v1.1_pubmed')
model = Bert_LSTM_opt(config, learning_rate=config.learning_rate, static_bert=False,
					  bert_name_or_path=biobert_path, bert_config=True)
model.load_weights(os.path.join(os.path.dirname(__file__), 'best_model/biobert_lstm_ft_28/model_weights.h5'))


@app.route('/')
def home():
	return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
	email = request.form.get('email')
	pmids = request.form.get('pmids_box')
	term = request.form.get('term_box')
	nmr_docs = request.form.get('term_box2')
	pmids_not_found = None
	docs_not_found = None
	if pmids:
		try:
			pmids = pmids.split(sep='\r\n')
			docs, pmids_not_found = pubmed_reader.pmids_to_docs(pmids, email, config)
			if not docs:
				return render_template('result.html', tables=None)
		except:
			return render_template('error.html', e=1)

	elif term:
		try:
			term = term.strip()
			nmr_docs = nmr_docs.strip()
			docs = pubmed_reader.term_to_docs(term, email, retmax=nmr_docs, config=config)
			if not docs:
				return render_template('result.html', tables=None)
		except:
			return render_template('error.html', e=2)


	elif request.files:
		try:
			updloaded_files = request.files.getlist('pdfs_box')
			docs, docs_not_found = pubmed_reader.pdfs_to_docs(updloaded_files, email, config)
			if not docs:
				return render_template('result.html', tables=None)
		except:
			return render_template('error_html', e=3)

	try:
		x_test_df = docs_to_pandasdocs(docs)

		x_test = Bert_preprocessing(x_test_df, config=config)

		yhat_probs = model.predict(x_test, verbose=0)

		yhat_probs = yhat_probs[:, 0]
		yhat_probs_copy = yhat_probs.copy()
		yhat_probs_df = pd.Series(data=yhat_probs_copy.reshape(len(x_test_df.index), ), index=x_test_df.index)
		yhat_probs_df[yhat_probs_df < 0.50] = - (1 - yhat_probs_df)
		sorted_yhat = yhat_probs_df.sort_values(ascending=False)
		ranked_df = sorted_yhat.to_frame().rename(columns={0: 'Conf. Score'})
		data = {'PMID': [], 'URL': []}
		for index in ranked_df.index.values:
			doc_title = x_test_df['Document'][index].raw_title
			data['PMID'].append(index)
			data['URL'].append('<a href="https://pubmed.ncbi.nlm.nih.gov/{0}/">{1}</a>'.format(index, doc_title))
		meta_df = pd.DataFrame(data, index=ranked_df.index)

		final_df = meta_df.join(ranked_df)
		if docs_not_found:
			data_not_found = {'PMID': [], 'URL': [], 'Conf. Score': []}
			for doc in docs_not_found:
				data_not_found['PMID'].append('Not Found')
				data_not_found['URL'].append(doc)
				data_not_found['Conf. Score'].append('NA')
			df_not_found = pd.DataFrame(data_not_found)
			final_df = final_df.append(df_not_found)

		elif pmids_not_found:
			data_not_found = {'PMID': [], 'URL': [], 'Conf. Score': []}
			for pmid in pmids_not_found:
				data_not_found['PMID'].append(pmid)
				data_not_found['URL'].append('Not Found')
				data_not_found['Conf. Score'].append('NA')
			df_not_found = pd.DataFrame(data_not_found)
			final_df = final_df.append(df_not_found)

		return render_template('result.html', tables=[final_df.to_html(escape=False, index=False, classes='data', )])
	except:
		render_template('error.html', e=4)


if __name__ == '__main__':
	# app.run(debug=True)
	from waitress import serve

	serve(app, host='0.0.0.0', port=5000)
