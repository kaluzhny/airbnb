import numpy as np
from sklearn.base import clone
from sklearn.cross_validation import StratifiedKFold
from xgboost.sklearn import XGBClassifier
from features import remove_sessions_columns
from scores import ndcg5_eval

n_folds = 4


def add_blend_feature(classifier, classes_count, remove_session_features,
                      x_train, y_train, x_test, train_columns, test_columns, feature_prefix,
                      random_state):
    # to_others = ['PT', 'AU', 'NL', 'DE', 'CA', 'ES', 'GB']
    # y_train = convert_outputs_to_others(y_train, to_others)
    #classes_count = len(le_.classes_) - len(to_others)

    no_sessions_classifiers, blend_train_no_session_feature = train_blend_no_session_feature(
        classifier, remove_session_features, x_train, y_train, train_columns, classes_count, random_state)

    x_train = np.hstack((x_train, blend_train_no_session_feature))

    if remove_session_features:
        x_test_no_sessions_features = remove_sessions_columns(x_test, train_columns)
    else:
        x_test_no_sessions_features = x_test

    blend_test_no_sessions = predict_blend_feature(x_test_no_sessions_features,
                                                   no_sessions_classifiers, classes_count)
    x_test = np.hstack((x_test, blend_test_no_sessions))

    for i in range(classes_count):
        feature_name = feature_prefix + str(i)
        train_columns = np.append(train_columns, [feature_name])
        test_columns = np.append(test_columns, [feature_name])

    return x_train, x_test, train_columns, test_columns


def train_blend_no_session_feature(classifier, remove_session_features,
                                   x_train, y_train, train_columns, classes_count,
                                   random_state):
    if remove_session_features:
        x_train_no_sessions_only_features = remove_sessions_columns(x_train, train_columns)
    else:
        x_train_no_sessions_only_features = x_train

    no_sessions_classifiers, blend_train_no_session_feature = train_blend_feature(
        classifier, x_train_no_sessions_only_features, y_train, classes_count, random_state)
    return no_sessions_classifiers, blend_train_no_session_feature


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
            classifier.fit(x_blend_train, y_blend_train, eval_metric=ndcg5_eval)
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

