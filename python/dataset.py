import numpy as np


class DataSet(object):

    @staticmethod
    def create_from_df(df):
        ids = df['id'].tolist()
        df = df.drop(['id'], axis=1)
        columns = df.columns.values.tolist()
        data = df.as_matrix().astype(float)
        return DataSet(ids, columns, data)

    def __init__(self, ids, columns, data):
        assert(len(ids) == data.shape[0])
        assert(len(columns) == data.shape[1])
        self.ids_ = ids
        self.columns_ = columns
        self.data_ = data

    def filter_columns(self, columns):
        idxs = [self.columns_.index(col) for col in columns]
        return DataSet(self.ids_, columns, self.data_[:, idxs])

    def append_horizontal(self, right):
        assert(self.ids_ == right.ids_)
        return DataSet(self.ids_, self.columns_ + right.columns_,
                       np.hstack((self.data_, right.data_)))
