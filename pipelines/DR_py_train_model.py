import sys
sys.path.append('../')
from wrappers.bioc_wrapper import bioc_to_docs, bioc_to_relevances
from wrappers.pandas_wrapper import relevances_to_pandas, docs_to_pandasdocs
from mlearning.features_generator_docs import generate_features_dr, tfidf, serialize_features, deserialize_features
from mlearning.ml_config import MLConfig
from mlearning.ml import train_model, predict_with_model, evaluate_model, deserialize_mlconfig, serialize_mlconfig
from sklearn.feature_selection import SelectKBest, chi2, f_classif, SelectPercentile
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelBinarizer

fp_train = '../datasets/PMtask_Triage_TrainingSet.xml'
docs_train = bioc_to_docs(fp_train)
relevances_train = bioc_to_relevances(fp_train, 'protein-protein')


df_train = docs_to_pandasdocs(docs_train)
#X_train = generate_features_dr(df_train)
#serialize_features(X_train, path='../tests/features/all_training_features.txt')
X_train = deserialize_features('../tests/features/all_training_features.txt').drop(columns='Label')

y_train = relevances_to_pandas(X_train, relevances_train)

path = 'models/default_svm_model_scaler_proba.txt'
clf = SVC(probability=True)
scaler = StandardScaler()
feature_selector = SelectPercentile(f_classif, percentile = 0.8)
ml_config = MLConfig(clf, path, scaler=scaler,
                     feature_selector=feature_selector)


evaluate_model(X_train, y_train , ml_config)

train_model(X_train, y_train, ml_config)

predict_with_model(X_train, y_train, ml_config)

ml_config.write_report()




