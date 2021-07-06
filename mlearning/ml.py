
from data_structures.token import Token
from preprocessing.ml_config import MLConfig
import pandas
import pickle
import numpy as np
from sklearn import metrics
from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report, confusion_matrix, log_loss


def evaluate_model(X_set, y_set, mlconfig, cv=10, scoring=None, n_jobs=-1, report_output=True):
    if mlconfig.feature_selector is not None:
        feat_select = mlconfig.feature_selector
        feat_select.fit_transform(X_set, y_set)
        X_set = X_set[X_set.columns.values[feat_select.get_support()]]
    
    if mlconfig.scaler is not None:
        scaled_array = mlconfig.scaler.fit_transform(X_set)
        X_set = pandas.DataFrame(scaled_array, index=X_set.index, columns=X_set.columns)

    clf = mlconfig.classifier
    
    print('Computing Cross Validation')
    scores = cross_validate(clf, X_set, y_set, cv=cv, scoring=scoring, n_jobs=n_jobs)
    
    if report_output:
        mlconfig.report.to_report['Cross Validation'] = str(scores) + '\n' + '_'*100 +'\n'
    return scores


def train_model(X_set, y_set, mlconfig, report_output=True):
    if mlconfig.feature_selector is not None:
        feat_select = mlconfig.feature_selector
        feat_select.fit_transform(X_set, y_set)
        X_set = X_set[X_set.columns.values[feat_select.get_support()]]

    if mlconfig.scaler is not None:
        scaled_array = mlconfig.scaler.fit_transform(X_set)
        X_set = pandas.DataFrame(scaled_array, index=X_set.index, columns=X_set.columns)

    clf = mlconfig.classifier
    print('Training Model')
    clf.fit(X_set, y_set)

    mlconfig.features_cols_names = X_set.columns

    if report_output:
        mlconfig.report.to_report['Features Used'] =  'Selector Params:' + str(mlconfig.feature_selector.get_params()) \
                                                      + '\n' + str(len(X_set.columns.values)) + '\n' \
                                                      + str(X_set.columns.values)

    if mlconfig.path is not None:
        serialize_config(mlconfig)


def predict_with_model(X_set, y_set, mlconfig, report_output=True, report_name=None):
    # for col in mlconfig.features_cols_names:
    #     if col not in X_set.columns:
    #         print('the column {} does not exist'.format(col))
    #         X_set[col] = X_set.mean()   #this value can change with the scaler??
    
    X_set = X_set[mlconfig.features_cols_names]

    if mlconfig.scaler is not None:
        scaled_array = mlconfig.scaler.transform(X_set)
        X_set = pandas.DataFrame(scaled_array, index=X_set.index, columns=X_set.columns)

    clf = mlconfig.classifier
    print('Predicting')
    y_pred = clf.predict(X_set)

    if report_output:
        cm = np.array2string(confusion_matrix(y_set, y_pred))
        cr = classification_report(y_set, y_pred)
        res = '\nConfusion Matrix\n  TN   FP\n{}\n  FN   TP  \n\n'.format(cm)
        res += 'Classification Report\n{}'.format(cr)
        if mlconfig.report.to_report['Training Prediction'] != '':
            mlconfig.report.to_report['Test Prediction'] = res + '\n' + '_'*100 +'\n'
        else:
            mlconfig.report.to_report['Training Prediction'] = res + '\n' + '_'*100 +'\n'
    
    if mlconfig.path is not None:
        serialize_config(mlconfig)

    y_pred_proba = -1
    if mlconfig.classifier.get_params()['probability'] is not False:
        y_pred_proba = predict_proba(X_set, mlconfig)

    indexes = list(X_set.index.values)
    data = {
        'Predicted': pandas.Series(y_pred, index=indexes),
        'Confidence': pandas.Series(y_pred_proba, index=indexes)
    }
    dataframe = pandas.DataFrame(data)
    return dataframe


def predict_proba(X_set, mlconfig):
    bi_y_pred_proba = mlconfig.classifier.predict_proba(X_set)
    y_pred_proba = []

    for document_bi_prob in bi_y_pred_proba:
        prob_label_1 = document_bi_prob[0]
        prob_label_2 = document_bi_prob[1]
        if prob_label_1 > prob_label_2:
            y_pred_proba.append(str(prob_label_1))
        else:
            y_pred_proba.append(str(prob_label_2))

    return y_pred_proba



def serialize_config(config):
    with open(config.path, 'wb') as fp:
        pickle.dump(config, fp)


def deserialize_config(path):
    with open(path, 'rb') as fp:
        config = pickle.load(fp)
    return config