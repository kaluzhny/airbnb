import numpy as np
import pickle


class DataSet(object):

    @staticmethod
    def create_from_df(df):
        ids = df['id'].tolist()
        df = df.drop(['id'], axis=1)
        columns = df.columns.values.tolist()
        data = df.as_matrix().astype(float)
        return DataSet(ids, columns, data)

    @staticmethod
    def save_to_file(ds, path):
        data = {
            'ids': ds.ids_,
            'columns': ds.columns_,
            'data': ds.data_
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    @staticmethod
    def load_from_file(path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return DataSet(data['ids'], data['columns'], data['data'])

    def __init__(self, ids, columns, data):
        assert(len(ids) == data.shape[0])
        assert(len(columns) == data.shape[1])
        self.ids_ = ids
        self.columns_ = columns
        self.data_ = data

    def filter_columns(self, columns):
        data = np.zeros((len(self.ids_), len(columns)))
        for i, col in enumerate(columns):
            try:
                idx = self.columns_.index(col)
                data[:, i] = self.data_[:, idx]
            except ValueError:
                pass

        return DataSet(self.ids_, columns, data)

    def filter_rows(self, idxs):
        ids = [self.ids_[idx] for idx in idxs]
        return DataSet(ids, self.columns_, self.data_[idxs,:])

    def append_horizontal(self, right):
        assert(self.ids_ == right.ids_)
        return DataSet(self.ids_, self.columns_ + right.columns_,
                       np.hstack((self.data_, right.data_)))
