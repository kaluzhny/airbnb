import math
import numpy as np
import pandas as pd
from collections import namedtuple
from datetime import date
from data import read_from_csv
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost.sklearn import XGBClassifier
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier,\
    GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier

from scores import ndcg_at_k, score, print_xgboost_scores
from features import make_one_hot, do_pca, str_to_date, remove_sessions_columns, remove_no_sessions_columns,\
    divide_by_has_sessions, sync_columns, add_sessions_features, print_columns, add_features
from probabilities import print_probabilities, correct_probs, adjust_test_data
from blend import add_blend_feature, train_blend_no_session_feature, train_blend_feature, predict_blend_feature


TaskCore = namedtuple('TaskCore', ['data_file', 'sessions_data_file', 'test_data_file', 'submission_file',
                                   'cv_ratio', 'n_threads', 'n_seed'])
destinations = ['NDF', 'US', 'other', 'FR', 'CA', 'GB', 'ES', 'IT', 'DE', 'NL',  'AU', 'PT',]

le_ = LabelEncoder()
le_.fit(destinations)


def x_from_df(data_df, sessions_df, is_test, test_columns=None):

    print('x_from_df <<')
    data_df = add_features(data_df, test_columns)
    data_df = add_sessions_features(data_df, sessions_df)
    data_df = data_df.drop([ 'id' ], axis=1)
    if not is_test:
        data_df = data_df.drop([ 'country_destination' ], axis=1)
    print_columns(data_df.columns.values)

    x = data_df.as_matrix()
    x = x.astype(float)
    print('x_from_df >>')
    return x, data_df.columns.values


def load_sessions(path):
    return pd.read_csv(path, usecols=['user_id', 'action', 'action_type', 'action_detail'])


class Task(object):
    def __init__(self, task_core):
        self.task_core = task_core

    def load_data(self):
        raise NotImplementedError("Implement this")

    def run(self):
        return self.load_data()


class TrainingDataTask(Task):
    def __init__(self, task_core, test_columns):
        super(TrainingDataTask, self).__init__(task_core)
        self.test_columns = test_columns

    def load_train_data(self, sessions_df):
        data_df = read_from_csv(self.task_core.data_file, self.task_core.n_seed
                                #, max_rows=10000
                                )
        x, columns = x_from_df(data_df, sessions_df, False, test_columns=self.test_columns)

        labels = data_df['country_destination'].values
        y = le_.transform(labels)

        return {'X': x, 'Y': y, 'columns': columns}

    def load_data(self):
        print('Loading train data ', self.task_core.data_file, ' to pandas data frame')
        sessions_df = load_sessions(self.task_core.sessions_data_file)
        data = self.load_train_data(sessions_df)
        return data


class TestDataTask(Task):

    def load_test_data(self, sessions_df):
        data_df = read_from_csv(self.task_core.test_data_file, self.task_core.n_seed
                                #, max_rows=10000
                                )
        x, columns = x_from_df(data_df, sessions_df, True)

        ids = data_df['id'].tolist()
        return {'X': x, 'columns': columns, 'ids': ids}

    def load_data(self):
        print('Loading test data ', self.task_core.test_data_file, ' to pandas data frame')
        sessions_df = load_sessions(self.task_core.sessions_data_file)
        data = self.load_test_data(sessions_df)
        return data


class CrossValidationScoreTask(Task):
    def __init__(self, task_core, classifier):
        super(CrossValidationScoreTask, self).__init__(task_core)
        self.classifier = classifier

    def load_data(self):
        test_ratio = self.task_core.cv_ratio

        test_data = TestDataTask(self.task_core).run()
        test_columns = test_data['columns']

        data = TrainingDataTask(self.task_core, test_columns).run()
        train_columns = data['columns']
        x = data['X']
        y = data['Y']

        x_sessions, y_sessions, x_no_sessions, y_no_sessions = divide_by_has_sessions(
            list(train_columns).index('s_count_all'), x, y)

        x_sessions_train, x_sessions_test, y_sessions_train, y_sessions_test = \
            train_test_split(x_sessions, y_sessions, test_size=test_ratio)
        # x_sessions_test, y_sessions_test = adjust_test_data(x_sessions_test, y_sessions_test, le_)

        print('x_sessions_train shape: ', x_sessions_train.shape)
        print('x_sessions_test shape: ', x_sessions_test.shape)
        print('x_no_sessions shape: ', x_no_sessions.shape)

        x_train = np.vstack((x_sessions_train, x_no_sessions))
        y_train = np.concatenate((y_sessions_train, y_no_sessions))

        print('Training no_session classifier...')

        no_sessions_classifiers, blend_train_no_session_feature = train_blend_no_session_feature(
            x_train, y_train, train_columns)
        blend_sessions_train_no_session_feature = blend_train_no_session_feature[:x_sessions_train.shape[0], :]
        print('blend_sessions_train_no_session_feature shape: ', blend_sessions_train_no_session_feature.shape)

        print('Training session classifier...')
        x_sessions_only_session_features_train = remove_no_sessions_columns(x_sessions_train, train_columns)
        print('x_sessions_only_session_features_train shape: ', x_sessions_only_session_features_train.shape)
        sessions_classifiers, blend_sessions_train_session_feature = train_blend_feature(
            XGBClassifier(objective='multi:softmax', max_depth=4, nthread=2),
            x_sessions_only_session_features_train, y_sessions_train, len(le_.classes_))

        x_sessions_train = np.hstack((x_sessions_train,
                                      blend_sessions_train_no_session_feature,
                                      blend_sessions_train_session_feature))

        print('x_sessions_train shape: ', x_sessions_train.shape)

        print('Training aggregate classifier...')
        self.classifier.fit(x_sessions_train, y_sessions_train, eval_metric='ndcg@5')
        print_xgboost_scores(self.classifier)

        print('Predicting no session feature for test')
        x_sessions_test_no_sessions_only_features = remove_sessions_columns(x_sessions_test, train_columns)
        blend_no_sessions_test = predict_blend_feature(x_sessions_test_no_sessions_only_features,
                                                       no_sessions_classifiers, len(le_.classes_))

        print('Predicting session feature for test')
        x_sessions_test_sessions_only_features = remove_no_sessions_columns(x_sessions_test, train_columns)
        blend_sessions_test = predict_blend_feature(x_sessions_test_sessions_only_features,
                                                    sessions_classifiers, len(le_.classes_))

        x_sessions_only_session_features_test = remove_no_sessions_columns(x_sessions_test, train_columns)
        x_sessions_test = np.hstack((x_sessions_test,
                                     blend_no_sessions_test,
                                     blend_sessions_test))

        print('x_sessions_test shape: ', x_sessions_test.shape)

        print('Predicting y for test')
        probabilities = self.classifier.predict_proba(x_sessions_test)
        # probabilities = correct_probs(probabilities, le_)
        s = score(probabilities, y_sessions_test, le_)
        return {'Score': s}


class MakePredictionTask(Task):
    def __init__(self, task_core, classifier):
        super(MakePredictionTask, self).__init__(task_core)
        self.classifier = classifier

    def load_data(self):

        classes_count = len(le_.classes_)

        # load test data
        test_data = TestDataTask(self.task_core).run()
        x_test = test_data['X']
        test_columns = test_data['columns']
        test_ids = test_data['ids']
        print('test_ids len: ', len(test_ids))

        # train
        train_data = TrainingDataTask(self.task_core, test_columns).run()
        x_train = train_data['X']
        y_train = train_data['Y']
        train_columns = train_data['columns']

        x_test, test_columns = sync_columns(x_test, test_columns, train_columns)

        print('printing test_columns')
        print_columns(test_columns)

        print('printing train_columns')
        print_columns(train_columns)

        print('adding xgboost feature 1...')
        x_train, x_test, train_columns, test_columns = add_blend_feature(
            XGBClassifier(objective='multi:softmax', max_depth=4,
                          nthread=self.task_core.n_threads, seed=self.task_core.n_seed),
            classes_count,
            True,
            x_train, y_train, x_test,
            train_columns, test_columns,
            "xg_1_", self.task_core.n_seed)

        print('adding rfc feature 1...')
        x_train, x_test, train_columns, test_columns = add_blend_feature(
           RandomForestClassifier(n_estimators=100, criterion='gini',
                                  n_jobs=self.task_core.n_threads, random_state=self.task_core.n_seed),
           classes_count,
           True,
           x_train, y_train, x_test,
           train_columns, test_columns,
           "rfc_1_", self.task_core.n_seed)
        x_train, x_test, train_columns, test_columns = add_blend_feature(
           RandomForestClassifier(n_estimators=100, criterion='entropy',
                                  n_jobs=self.task_core.n_threads, random_state=self.task_core.n_seed),
           classes_count,
           True,
           x_train, y_train, x_test,
           train_columns, test_columns,
           "rfc_2_", self.task_core.n_seed)

        print('adding etc feature 1...')
        x_train, x_test, train_columns, test_columns = add_blend_feature(
            ExtraTreesClassifier(n_estimators=100, criterion='gini',
                                 n_jobs=self.task_core.n_threads, random_state=self.task_core.n_seed),
            classes_count,
            True,
            x_train, y_train, x_test,
            train_columns, test_columns,
            "etc_1_", self.task_core.n_seed)
        x_train, x_test, train_columns, test_columns = add_blend_feature(
            ExtraTreesClassifier(n_estimators=100, criterion='entropy',
                                 n_jobs=self.task_core.n_threads, random_state=self.task_core.n_seed),
            classes_count,
            True,
            x_train, y_train, x_test,
            train_columns, test_columns,
            "etc_2_", self.task_core.n_seed)

        print('adding gbc feature 1...')
        x_train, x_test, train_columns, test_columns = add_blend_feature(
            GradientBoostingClassifier(max_depth=4, n_estimators=100, random_state=self.task_core.n_seed),
            classes_count,
            True,
            x_train, y_train, x_test,
            train_columns, test_columns,
            "gbc_1_", self.task_core.n_seed)

        print('adding ada feature 1...')
        x_train, x_test, train_columns, test_columns = add_blend_feature(
            AdaBoostClassifier(n_estimators=300, random_state=self.task_core.n_seed),
            classes_count,
            True,
            x_train, y_train, x_test,
            train_columns, test_columns,
            "ada_1_", self.task_core.n_seed)

        # print('adding sgb feature 1...')
        # x_train, x_test, train_columns, test_columns = add_blend_feature(
        #     SGDClassifier(n_jobs=2),
        #     classes_count,
        #     True,
        #     x_train, y_train, x_test,
        #     train_columns, test_columns,
        #     "sgd_1_", self.task_core.n_seed)

        print('x_train: ', x_train.shape)
        print('x_test: ', x_test.shape)
        print('dividing train/test sets...')
        x_train_sessions, y_train_sessions, x_train_no_sessions, y_train_no_sessions = divide_by_has_sessions(
            # list(train_columns).index('s_count_all'),
            list(train_columns).index('year_first_active'),
            x_train, y_train)

        print('x_train_sessions: ', x_train_sessions.shape)
        print('x_train_no_sessions: ', x_train_no_sessions.shape)

        x_test_sessions = x_test
        train_columns_2 = train_columns
        test_columns_2 = test_columns

        print('adding xgboost feature 2...')
        x_train_sessions, x_test_sessions, train_columns_2, test_columns_2 = add_blend_feature(
            XGBClassifier(objective='multi:softmax', max_depth=4,
                          nthread=self.task_core.n_threads, seed=self.task_core.n_seed),
            classes_count,
            False,
            x_train_sessions, y_train_sessions, x_test_sessions,
            train_columns_2, test_columns_2,
            "xg_2014_", self.task_core.n_seed)

        print('adding rfc feature 2...')
        x_train_sessions, x_test_sessions, train_columns_2, test_columns_2 = add_blend_feature(
            RandomForestClassifier(n_estimators=100, criterion='gini',
                                   n_jobs=self.task_core.n_threads, random_state=self.task_core.n_seed),
            classes_count,
            False,
            x_train_sessions, y_train_sessions, x_test_sessions,
            train_columns_2, test_columns_2,
            "rfc_2014_1_", self.task_core.n_seed)

        x_train_sessions, x_test_sessions, train_columns_2, test_columns_2 = add_blend_feature(
            RandomForestClassifier(n_estimators=100, criterion='entropy',
                                   n_jobs=self.task_core.n_threads, random_state=self.task_core.n_seed),
            classes_count,
            False,
            x_train_sessions, y_train_sessions, x_test_sessions,
            train_columns_2, test_columns_2,
            "rfc_2014_2_", self.task_core.n_seed)

        print('adding etc feature 2...')
        x_train_sessions, x_test_sessions, train_columns_2, test_columns_2 = add_blend_feature(
            ExtraTreesClassifier(n_estimators=100, criterion='gini',
                                 n_jobs=self.task_core.n_threads, random_state=self.task_core.n_seed),
            classes_count,
            False,
            x_train_sessions, y_train_sessions, x_test_sessions,
            train_columns_2, test_columns_2,
            "etc_2014_1_", self.task_core.n_seed)

        x_train_sessions, x_test_sessions, train_columns_2, test_columns_2 = add_blend_feature(
            ExtraTreesClassifier(n_estimators=100, criterion='entropy',
                                 n_jobs=self.task_core.n_threads, random_state=self.task_core.n_seed),
            classes_count,
            False,
            x_train_sessions, y_train_sessions, x_test_sessions,
            train_columns_2, test_columns_2,
            "etc_2014_2_", self.task_core.n_seed)

        print('adding gbc feature 2...')
        x_train_sessions, x_test_sessions, train_columns_2, test_columns_2 = add_blend_feature(
            GradientBoostingClassifier(max_depth=4, n_estimators=100, random_state=self.task_core.n_seed),
            classes_count,
            False,
            x_train_sessions, y_train_sessions, x_test_sessions,
            train_columns_2, test_columns_2,
            "gbc_2014_1_", self.task_core.n_seed)

        print('adding ada feature 2...')
        x_train_sessions, x_test_sessions, train_columns_2, test_columns_2 = add_blend_feature(
            AdaBoostClassifier(n_estimators=300, random_state=self.task_core.n_seed),
            classes_count,
            False,
            x_train_sessions, y_train_sessions, x_test_sessions,
            train_columns_2, test_columns_2,
            "ada_2014_1_", self.task_core.n_seed)

        # print('adding sgb feature 2...')
        # x_train_sessions, x_test_sessions, train_columns_2, test_columns_2 = add_blend_feature(
        #     SGDClassifier(n_jobs=2),
        #     classes_count,
        #     False,
        #     x_train_sessions, y_train_sessions, x_test_sessions,
        #     train_columns_2, test_columns_2,
        #     "sgb_2014_1_", self.task_core.n_seed)

        print('Predicting all features...')
        print_columns(train_columns_2)
        probabilities = simple_predict(clone(self.classifier), x_train_sessions,
                                       y_train_sessions, x_test_sessions,
                                       train_columns_2)
        print_probabilities(probabilities)

        save_submission(test_ids, probabilities, self.task_core.submission_file)


def convert_outputs_to_others(y, other_labels):
    def convert_func(val):
        label = le_.inverse_transform([val])[0]
        if label in other_labels:
            return le_.transform(['other'])[0]
        else:
            return val
    return np.vectorize(convert_func)(y)





def simple_predict(classifier, x_train, y_train, x_test, columns=None):
    print('x_train shape: ', x_train.shape)
    print('x_test shape: ', x_test.shape)
    classifier.fit(x_train, y_train, eval_metric='ndcg@5')
    print_xgboost_scores(classifier, columns)
    probabilities = classifier.predict_proba(x_test)
    return probabilities


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

