import sys
sys.path.append('../')
from src.biotmpy.wrappers.bioc_wrapper import bioc_to_docs, bioc_to_relevances
from src.biotmpy.wrappers.pandas_wrapper import relevances_to_pandas, docs_to_pandasdocs, pandasdocs_to_docs
from src.biotmpy.mlearning import generate_features_dr, tfidf, serialize_features, deserialize_features
from src.biotmpy.mlearning.ml import train_model, predict_with_model, evaluate_model, deserialize_mlconfig, serialize_mlconfig

fp_test = '../data/PMtask_Triage_TestSet.xml'
docs_test = bioc_to_docs(fp_test)
relevances_test = bioc_to_relevances(fp_test, 'protein-protein')

df_test = docs_to_pandasdocs(docs_test)
# df_test_features = generate_features_dr(df_test)
# serialize_features(df_test_features, path='../tests/features/all_test_features.txt')
X_test = deserialize_features(path='../tests/features/all_test_features.txt').drop(columns='Label')

y_test = relevances_to_pandas(X_test, relevances_test)

svm_config = deserialize_mlconfig('models/default_svm_model_scaler_xtrain.txt')

df_pred = predict_with_model(X_test, y_test, svm_config)

# svm_config.write_report()

# predicted_relevances = pandas_to_relevances(df_pred)
# docs_to_biocdocs(predicted_docs, 'new.BioC.xml')
# relevances_to_bioc(predicted_relevances, 'new.BioC.xml')
