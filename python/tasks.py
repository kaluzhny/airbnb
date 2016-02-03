import os.path
import numpy as np
import pandas as pd
from collections import namedtuple
from data import read_from_csv
from sklearn.cross_validation import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from xgboost.sklearn import XGBClassifier
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier,\
    GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer

from sklearn.ensemble import BaggingClassifier

from scores import ndcg_at_k, score, print_xgboost_scores, ndcg5_eval
from features import make_one_hot, do_pca, str_to_date, remove_sessions_columns, remove_no_sessions_columns,\
    divide_by_has_sessions, sync_columns, sync_columns_2, add_sessions_features, print_columns, add_features,\
    add_tsne_features, get_one_hot_columns
from probabilities import print_probabilities, correct_probs, adjust_test_data
from blend import train_blend_feature, predict_blend_feature, simple_predict, get_blend_features
from dataset import DataSet


TaskCore = namedtuple('TaskCore', ['data_file', 'sessions_data_file', 'test_data_file', 'submission_file',
                                   'cv_ratio', 'n_threads', 'n_seed', 'cache_dir'])
destinations = ['NDF', 'US', 'other', 'FR', 'CA', 'GB', 'ES', 'IT', 'DE', 'NL',  'AU', 'PT',]

le_ = LabelEncoder()
le_.fit(destinations)


def ds_from_df(data_df, sessions_df, is_test):
    print('ds_from_df <<')
    data_df = add_features(data_df)
    data_df = add_sessions_features(data_df, sessions_df)
    if not is_test:
        data_df = data_df.drop(['country_destination'], axis=1)
    print('ds_from_df >>')
    return DataSet.create_from_df(data_df)


def load_sessions(path):
    return pd.read_csv(path, usecols=['user_id', 'action', 'action_type', 'action_detail', 'device_type', 'secs_elapsed'])


class Task(object):
    def __init__(self, task_core):
        self.task_core = task_core

    def load_data(self):
        raise NotImplementedError("Implement this")

    def run(self):
        return self.load_data()


class TrainingDataTask(Task):
    def __init__(self, task_core):
        super(TrainingDataTask, self).__init__(task_core)

    def load_train_data(self, sessions_df):
        data_df = read_from_csv(self.task_core.data_file, self.task_core.n_seed
                                #, max_rows=50000
                                )

        cache_file = os.path.join(self.task_core.cache_dir, 'features_train_' + str(len(data_df.index)) + '.p')
        if os.path.isfile(cache_file):
            print('Loading train features from file')
            x = DataSet.load_from_file(cache_file)
        else:
            x = ds_from_df(data_df, sessions_df, False)
            print('saving train features to file')
            DataSet.save_to_file(x, cache_file)

        labels = data_df['country_destination'].values
        y = le_.transform(labels)
        return x, y

    def load_data(self):
        print('Loading train data ', self.task_core.data_file, ' to pandas data frame')
        sessions_df = load_sessions(self.task_core.sessions_data_file)
        data = self.load_train_data(sessions_df)
        return data


class TestDataTask(Task):
    def load_test_data(self, sessions_df):
        data_df = read_from_csv(self.task_core.test_data_file, self.task_core.n_seed
                                #, max_rows=50000
                                )

        cache_file = os.path.join(self.task_core.cache_dir, 'features_test_' + str(len(data_df.index)) + '.p')
        if os.path.isfile(cache_file):
            print('Loading test features from file')
            x = DataSet.load_from_file(cache_file)
        else:
            x = ds_from_df(data_df, sessions_df, True)
            print('saving test features to file')
            DataSet.save_to_file(x, cache_file)

        return x

    def load_data(self):
        print('Loading test data ', self.task_core.test_data_file, ' to pandas data frame')
        sessions_df = load_sessions(self.task_core.sessions_data_file)
        return self.load_test_data(sessions_df)


def get_model_classifiers(n_threads, n_seed):

    classifiers_session_data = [
        # (MultinomialNB(), True, False, 'nb'),
        # (LogisticRegression(), False, False, 'lr'),
        (XGBClassifier(objective='multi:softprob', max_depth=4, n_estimators=100, learning_rate=0.1, nthread=n_threads, seed=n_seed), False, False, 'xg4softprob100_all'),
        (RandomForestClassifier(n_estimators=200, criterion='gini', n_jobs=n_threads, random_state=n_seed), False, False, 'rfc200_all'),
        (ExtraTreesClassifier(n_estimators=200, criterion='gini', n_jobs=n_threads, random_state=n_seed), False, False, 'etc200_all'),
        # (AdaBoostClassifier(n_estimators=100, random_state=n_seed), False, False, 'ada100'),
    ]

    classifiers_no_session_data = [
        # (MultinomialNB(), True, False, 'nb'),
        # (LogisticRegression(), False, False, 'lr'),
        (XGBClassifier(objective='multi:softprob', max_depth=4, n_estimators=100, learning_rate=0.1, nthread=n_threads, seed=n_seed), False, False, 'xg4softprob100'),
        (RandomForestClassifier(n_estimators=200, criterion='gini', n_jobs=n_threads, random_state=n_seed), False, False, 'rfc200'),
        (ExtraTreesClassifier(n_estimators=200, criterion='gini', n_jobs=n_threads, random_state=n_seed), False, False, 'etc200'),
        # (AdaBoostClassifier(n_estimators=100, random_state=n_seed), False, False, 'ada100'),
    ]

    classifiers_2014 = [
        (RandomForestClassifier(n_estimators=200, criterion='entropy', n_jobs=n_threads, random_state=n_seed), False, False, 'rfc200_e_2014'),
        (ExtraTreesClassifier(n_estimators=200, criterion='entropy', n_jobs=n_threads, random_state=n_seed), False, False, 'etc200_e_2014'),
    ]

    return classifiers_session_data, classifiers_no_session_data, classifiers_2014


def run_model(x_train, y_train, x_test, classes_count, classifier, n_threads, n_seed):
    print('printing train_columns/test_columns')
    assert(x_train.columns_ == x_test.columns_)
    print_columns(x_train.columns_)

    #x_train, x_test = add_tsne_features(x_train, x_test)

    classifiers_session_data, classifiers_no_session_data, classifiers_2014 = get_model_classifiers(n_threads, n_seed)
    y_train_3out = convert_outputs_to_others(y_train, ['FR', 'CA', 'GB', 'ES', 'IT', 'PT', 'NL', 'DE', 'AU'])
    # session_features_3out_knn_train, session_features_3out_knn_test = get_blend_features(
    #     [
    #         (KNeighborsClassifier(n_neighbors=2, n_jobs=n_threads), False, True, 'knn_2_3'),
    #         (KNeighborsClassifier(n_neighbors=4, n_jobs=n_threads), False, True, 'knn_4_3'),
    #         (KNeighborsClassifier(n_neighbors=16, n_jobs=n_threads), False, True, 'knn_16_3'),
    #         (KNeighborsClassifier(n_neighbors=64, n_jobs=n_threads), False, True, 'knn_64_3'),
    #         (KNeighborsClassifier(n_neighbors=256, n_jobs=n_threads), False, True, 'knn_256_3'),
    #         (KNeighborsClassifier(n_neighbors=512, n_jobs=n_threads), False, True, 'knn_512_3')
    #     ],
    #     3,
    #     remove_sessions_columns(x_train), y_train_3out,
    #     remove_sessions_columns(x_test),
    #     n_seed)
    # x_train = x_train.append_horizontal(session_features_3out_knn_train)
    # x_test = x_test.append_horizontal(session_features_3out_knn_test)

    session_features_3out_train, session_features_3out_test = get_blend_features(
        classifiers_session_data,
        3,
        x_train, y_train_3out,
        x_test,
        n_seed)
    x_train = x_train.append_horizontal(session_features_3out_train)
    x_test = x_test.append_horizontal(session_features_3out_test)

    no_session_features_train, no_session_features_test = get_blend_features(
        classifiers_no_session_data,
        classes_count,
        remove_sessions_columns(x_train), y_train,
        remove_sessions_columns(x_test),
        n_seed)
    x_train = x_train.append_horizontal(no_session_features_train)
    x_test = x_test.append_horizontal(no_session_features_test)

    # use 2014 only for final training
    x_train, y_train, _, _ = divide_by_has_sessions(
        x_train, y_train)

    features_2014_train, features_2014_test = get_blend_features(
        classifiers_no_session_data,
        classes_count,
        x_train, y_train,
        x_test,
        n_seed)
    x_train = x_train.append_horizontal(features_2014_train)
    x_test = x_test.append_horizontal(features_2014_test)

    xgb_classifier = XGBClassifier(objective='multi:softprob', nthread=n_threads, seed=n_seed)
    search_classifier = GridSearchCV(
        xgb_classifier,
        {
            'max_depth': [3, 4, 5],
        },
        cv=10, #n_iter=10,
        verbose=10,
        n_jobs=1,
        scoring=make_scorer((lambda true_values, predictions: score(predictions, true_values)), needs_proba=True)
    )
    x_search = x_train.data_
    y_search = y_train
    perm = np.random.permutation(x_search.shape[0])
    x_search = x_search[perm,:]
    y_search = y_search[perm]
    search_classifier.fit(x_search, y_search)
    print('grid_scores_: ', search_classifier.grid_scores_)
    print('best_score_: ', search_classifier.best_score_)
    print('best_params_: ', search_classifier.best_params_)

    xgb = XGBClassifier(objective='multi:softprob', learning_rate=0.1, max_depth=4, nthread=n_threads, seed=n_seed)
    bag = BaggingClassifier(base_estimator=xgb, n_estimators=50, random_state=n_seed, verbose=10)
    probabilities = simple_predict(bag, x_train, y_train, x_test)

    return probabilities


class CrossValidationScoreTask(Task):
    def __init__(self, task_core, classifier):
        super(CrossValidationScoreTask, self).__init__(task_core)
        self.classifier = classifier

    def load_data(self):
        classes_count = len(le_.classes_)

        # load test data
        # x_test = TestDataTask(self.task_core).run()

        # train
        x_train, y_train = TrainingDataTask(self.task_core).run()

        # split
        train_idxs, test_idxs = list(StratifiedShuffleSplit(y_train, 1, test_size=self.task_core.cv_ratio,
                                                       random_state=self.task_core.n_seed))[0]
        x_test = x_train.filter_rows_by_idxs(test_idxs)
        y_test = y_train[test_idxs]
        x_train = x_train.filter_rows_by_idxs(train_idxs)
        y_train = y_train[train_idxs]

        # 2014 only for test
        x_test, y_test, _, _ = divide_by_has_sessions(x_test, y_test)

        print('running prediction model')
        probabilities = run_model(x_train, y_train, x_test, classes_count, self.classifier,
                                  self.task_core.n_threads, self.task_core.n_seed)

        print_probabilities(probabilities)
        s = score(probabilities, y_test)
        return {'Score': s}


class MakePredictionTask(Task):
    def __init__(self, task_core, classifier):
        super(MakePredictionTask, self).__init__(task_core)
        self.classifier = classifier

    def load_data(self):

        classes_count = len(le_.classes_)

        # load test data
        x_test = TestDataTask(self.task_core).run()

        # train
        x_train, y_train = TrainingDataTask(self.task_core).run()
        # perm_idxs = list(np.random.permutation(y_train.shape[0]))
        # x_train = x_train.filter_rows(perm_idxs)
        # y_train = y_train[perm_idxs]

        x_test, x_train = sync_columns_2(x_test, x_train)

        probabilities = run_model(x_train, y_train, x_test, classes_count, self.classifier,
                                  self.task_core.n_threads, self.task_core.n_seed)

        print_probabilities(probabilities)

        save_submission(x_test.ids_, probabilities, self.task_core.submission_file)


def convert_outputs_to_others(y, other_labels):
    def convert_func(val):
        label = le_.inverse_transform([val])[0]
        if label in other_labels:
            return le_.transform(['other'])[0]
        else:
            return val
    return np.vectorize(convert_func)(y)


def save_submission(ids, probabilities, path):
    submission_rows = ['id,country']
    print('Generating submission...')
    for user_idx, user_id in enumerate(ids):
        predicted_destinations = le_.inverse_transform(np.argsort(probabilities[user_idx]))[::-1][:5]
        for destination_idx in range(5):
            destination = predicted_destinations[destination_idx]
            submission_rows.append(user_id + ',' + destination)

    with open(path, 'w') as f:
        print('\n'.join(submission_rows), end="", file=f)

