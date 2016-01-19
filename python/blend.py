import numpy as np
from sklearn.base import clone
from sklearn.cross_validation import StratifiedKFold
from xgboost.sklearn import XGBClassifier
from features import remove_sessions_columns
from dataset import DataSet

n_folds = 4


def get_blend_feature(classifier, classes_count,
                      x_train, y_train, x_test, feature_prefix,
                      random_state):
    assert(x_train.columns_ == x_test.columns_)

    classifiers, feature_train_data = train_blend_feature(classifier, x_train.data_, y_train, classes_count, random_state)
    feature_test_data = predict_blend_feature(x_test.data_, classifiers, classes_count)
    new_columns = [feature_prefix + str(i) for i in range(classes_count)]

    return DataSet(x_train.ids_, new_columns, feature_train_data),\
           DataSet(x_test.ids_, new_columns, feature_test_data)


def add_blend_feature(classifiers, classes_count, remove_session_features,
                      x_train, y_train, x_test, random_state):

    x_train_featured = x_train
    x_test_featured = x_test

    if remove_session_features:
        x_train = remove_sessions_columns(x_train)
        x_test = remove_sessions_columns(x_test)

    for classifier, classifier_name in classifiers:
        print('adding feature with classifier ' + classifier_name + ' ...')
        feature_prefix = classifier_name + '_'
        feature_train, feature_test = get_blend_feature(classifier, classes_count,
                                                        x_train, y_train, x_test, feature_prefix,
                                                        random_state)
        x_train_featured = x_train_featured.append_horizontal(feature_train)
        x_test_featured = x_test_featured.append_horizontal(feature_test)

    return x_train_featured, x_test_featured


def train_blend_feature(classifier, x, y, classes_count, random_state):
    no_sessions_classifiers = [clone(classifier) for i in range(n_folds)]

    print('train_blend_feature: x - ', x.shape, '; y - ', y.shape)

    folds = list(StratifiedKFold(y, n_folds, random_state=random_state))
    blend_train = np.zeros((x.shape[0], classes_count))
    for i, (train_idx, test_idx) in enumerate(folds):
        print('fold: ', i)
        x_blend_train = x[train_idx]
        y_blend_train = y[train_idx]
        x_blend_test = x[test_idx]

        classifier = no_sessions_classifiers[i]
        if isinstance(classifier, XGBClassifier):
            classifier.fit(x_blend_train, y_blend_train, eval_metric='ndcg@5')
        else:
            classifier.fit(x_blend_train, y_blend_train)
        y_blend_predicted = no_sessions_classifiers[i].predict_proba(x_blend_test)
        blend_train[test_idx, :classes_count] = y_blend_predicted
    print('blend_train shape: ', blend_train.shape)

    return no_sessions_classifiers, blend_train


def predict_blend_feature(x, classifiers, classes_count):
    blend_test_all = np.zeros((x.shape[0], n_folds, classes_count))
    for i in range(n_folds):
        blend_test_all[:, i, :] = classifiers[i].predict_proba(x)
    return blend_test_all.mean(1)
