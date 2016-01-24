import numpy as np
from sklearn.base import clone
from sklearn.cross_validation import StratifiedKFold
from xgboost.sklearn import XGBClassifier
from features import remove_sessions_columns
from dataset import DataSet
from scores import ndcg5_eval, print_xgboost_scores, score
from sklearn.preprocessing import StandardScaler


n_folds = 4


def get_blend_feature(classifier, scale, classes_count,
                      x_train, y_train, x_test, feature_prefix,
                      random_state):
    assert(x_train.columns_ == x_test.columns_)

    train_blend_feature_result = train_blend_feature(classifier, scale, x_train.data_, y_train, classes_count, random_state)
    if scale:
        classifiers, scalers, feature_train_data = train_blend_feature_result
    else:
        classifiers, feature_train_data = train_blend_feature_result
        scalers = None

    feature_test_data = predict_blend_feature(x_test.data_, classifiers, scalers, classes_count)
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

    for classifier, scale, classifier_name in classifiers:
        print('adding feature with classifier ' + classifier_name + ' ...')
        feature_prefix = classifier_name + '_'
        feature_train, feature_test = get_blend_feature(classifier, scale, classes_count,
                                                        x_train, y_train, x_test, feature_prefix,
                                                        random_state)
        x_train_featured = x_train_featured.append_horizontal(feature_train)
        x_test_featured = x_test_featured.append_horizontal(feature_test)

    return x_train_featured, x_test_featured


def train_blend_feature(classifier, scale, x, y, classes_count, random_state):
    classifiers = [clone(classifier) for i in range(n_folds)]

    if scale:
        scalers = [StandardScaler() for i in range(n_folds)]
    else:
        scalers = None

    print('train_blend_feature: scale - ', scale, ', x - ', x.shape, '; y - ', y.shape)

    scores = []

    folds = list(StratifiedKFold(y, n_folds, shuffle=True, random_state=random_state))
    blend_train = np.zeros((x.shape[0], classes_count))
    for i, (train_idx, test_idx) in enumerate(folds):
        print('fold: ', i)
        x_blend_train = x[train_idx]
        y_blend_train = y[train_idx]
        x_blend_test = x[test_idx]

        classifier = classifiers[i]

        if scale:
            scaler = scalers[i].fit(x_blend_train)
            x_blend_train = scaler.transform(x_blend_train)
            x_blend_test = scaler.transform(x_blend_test)

        if isinstance(classifier, XGBClassifier):
            classifier.fit(x_blend_train, y_blend_train, eval_metric=ndcg5_eval) # 'ndcg@5')
        else:
            classifier.fit(x_blend_train, y_blend_train)
        y_blend_predicted = classifiers[i].predict_proba(x_blend_test)
        blend_train[test_idx, :classes_count] = y_blend_predicted

        # score
        y_blend_test = y[test_idx]
        scores.append(score(y_blend_predicted, y_blend_test))

    print('feature score: ', np.average(scores))

    if scale:
        return classifiers, scalers, blend_train
    else:
        return classifiers, blend_train


def predict_blend_feature(x, classifiers, scalers, classes_count):
    blend_test_all = np.zeros((x.shape[0], n_folds, classes_count))
    for i in range(n_folds):
        if scalers is not None:
            x = scalers[i].transform(x)
        blend_test_all[:, i, :] = classifiers[i].predict_proba(x)
    return blend_test_all.mean(1)


def simple_predict(classifier, x_train, y_train, x_test):
    x_train_data = x_train.data_
    x_test_data = x_test.data_
    print('x_train shape: ', x_train_data.shape)
    print('x_test shape: ', x_test_data.shape)


    # classifier.fit(x_train_data, y_train, eval_metric='ndcg@5')

    if isinstance(classifier, XGBClassifier):
        classifier.fit(x_train_data, y_train, eval_metric=ndcg5_eval)
    else:
        classifier.fit(x_train_data, y_train)

    # print_xgboost_scores(classifier, x_train.columns_)
    probabilities = classifier.predict_proba(x_test_data)
    return probabilities
