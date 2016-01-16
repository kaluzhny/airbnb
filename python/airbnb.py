import json
import os.path
import numpy as np
import sys

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tasks import TaskCore, CrossValidationScoreTask, MakePredictionTask
from xgboost.sklearn import XGBClassifier
from stdout_with_time import StdOutWithTime

sys.stdout = StdOutWithTime(sys.stdout)

n_threads = int(sys.argv[1])
n_seed = int(sys.argv[2])
submission_suffix = sys.argv[3]

print('running script with parameters: ',
      'n_threads - ', n_threads, '; ',
      'n_seed - ', n_seed, '; ',
      'submission_suffix - ' + submission_suffix)


def run_airbnb(target):

    with open('settings.json') as f:
        settings = json.load(f)

    data_dir = str(settings['competition-data-dir'])
    submission_dir = str(settings['submission-dir'])

    classifier = XGBClassifier(objective='multi:softmax', max_depth=4, nthread=n_threads, seed=n_seed)
    # (LogisticRegression(), 'lr'),
    # (XGBClassifier(objective='multi:softmax', max_depth=8, subsample=0.7, colsample_bytree=0.8, seed=0), 'air'),
    # (XGBClassifier(objective='multi:softmax', nthread=2, max_depth=4, learning_rate=0.03, n_estimators=10, subsample=0.5, colsample_bytree=0.5, seed=0), 'xg3'),
    # (RandomForestClassifier(n_estimators=100, min_samples_split=1, bootstrap=False, n_jobs=4, random_state=0), 'rf100mss1Bfrs0'),
    # (AdaBoostClassifier(base_estimator=None, n_estimators=50, learning_rate=1.0, algorithm='SAMME.R', random_state=None), 'ada'),
    # (AdaBoostClassifier(base_estimator=None, n_estimators=300, learning_rate=1.0, algorithm='SAMME.R', random_state=None), 'adaEst300'),
    # (SVC(C=1.0, kernel='rbf', degree=3, gamma=0.0, coef0=0.0, shrinking=True, probability=True, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, random_state=None), 'svc'),

    train_file = os.path.join(data_dir, 'train_users.csv')
    test_file = os.path.join(data_dir, 'test_users.csv')
    sessions_file = os.path.join(data_dir, 'sessions.csv')

    submission_file = os.path.join(
        submission_dir,
        'submission_simple_more_session_features_xg_etc_rf_ada_gbc_act120_det60_' +
        submission_suffix + '_seed_' + str(n_seed) + '.csv')

    def do_cross_validation():
            print('===== Making cross-validation')
            scores = []
            for i in range(5):
                task_core = TaskCore(data_file=train_file, sessions_data_file=sessions_file, test_data_file=test_file,
                                     submission_file=submission_file, cv_ratio=0.5,
                                     n_threads=n_threads, n_seed=n_seed)
                data = CrossValidationScoreTask(task_core, classifier).run()
                score = data['Score']
                scores.append(score)
                print('CV (', i, '): ', score)
            print('CV (mean): ', np.mean(scores))

    def make_prediction():
        task_core = TaskCore(data_file=train_file, sessions_data_file=sessions_file, test_data_file=test_file,
                             submission_file=submission_file, cv_ratio=0.5,
                             n_threads=n_threads, n_seed=n_seed)
        MakePredictionTask(task_core, classifier).run()

    if target == 'cv':
        do_cross_validation()
    elif target == 'prediction':
        make_prediction()

run_airbnb('prediction')
