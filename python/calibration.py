import json
import os.path
import numpy as np
import pandas as pd
from scores import ndcg_at_k


with open('settings.json') as f:
    settings = json.load(f)

data_dir = str(settings['competition-data-dir'])

train_df = pd.read_csv(os.path.join(data_dir, 'train_users.csv'))
test_df = pd.read_csv(os.path.join(data_dir, 'test_users.csv'))
#print(ndcg_at_k(np.asarray([0.0])))


submission_dir = str(settings['submission-dir'])
submission_file = os.path.join(submission_dir, 'calibration', 'submission_pt_only.csv')


submission_rows = ['id,country']

ids = test_df['id']
for user_idx, user_id in enumerate(ids):
    submission_rows.append(user_id + ',' + 'PT')

with open(submission_file, 'w') as f:
    print('\n'.join(submission_rows), end="", file=f)
