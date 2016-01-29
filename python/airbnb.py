import json
import os.path
import numpy as np
import sys
from random import randint

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
    cache_dir = str(settings['cache-dir'])

    classifier = XGBClassifier(objective='multi:softprob', max_depth=4, nthread=n_threads, seed=n_seed)
    # classifier = LogisticRegression()

    train_file = os.path.join(data_dir, 'train_users.csv')
    test_file = os.path.join(data_dir, 'test_users.csv')
    sessions_file = os.path.join(data_dir, 'sessions.csv')

    submission_file = os.path.join(
        submission_dir,
        'submission_xg_1feature_softprob4' +
        submission_suffix + '_seed_' + str(n_seed) + '.csv')

    def do_cross_validation():
            print('===== Making cross-validation')
            scores = []
            for i in range(10):
                seed = randint(0, 1000000)
                task_core = TaskCore(data_file=train_file, sessions_data_file=sessions_file, test_data_file=test_file,
                                     submission_file=submission_file, cv_ratio=0.5,
                                     n_threads=n_threads, n_seed=seed, cache_dir=cache_dir)
                data = CrossValidationScoreTask(task_core, classifier).run()
                score = data['Score']
                scores.append(score)
                print('CV (', i, '): ', score)
            print('CV (mean): ', np.mean(scores))

    def make_prediction():
        task_core = TaskCore(data_file=train_file, sessions_data_file=sessions_file, test_data_file=test_file,
                             submission_file=submission_file, cv_ratio=0.5,
                             n_threads=n_threads, n_seed=n_seed, cache_dir=cache_dir)
        MakePredictionTask(task_core, classifier).run()

    if target == 'cv':
        do_cross_validation()
    elif target == 'prediction':
        make_prediction()

run_airbnb('cv')
