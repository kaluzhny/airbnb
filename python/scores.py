import numpy as np
import operator


def dcg_at_k(r, k):
    return np.sum(r / np.log2(np.arange(2, r.size + 2)))


def ndcg_at_k(r, k=5):
    dcg_max = dcg_at_k(np.asfarray(sorted(r, reverse=True)), k)
    if not dcg_max:
        return 0.0
    return dcg_at_k(r, k) / dcg_max


def ndcg5_eval(predictions, true_values):
    labels = true_values.get_label()
    top = []
    for i in range(predictions.shape[0]):
        top.append(np.argsort(predictions[i])[::-1][:5])
    mat = np.reshape(np.repeat(labels, np.shape(top)[1]) == np.array(top).ravel(),np.array(top).shape).astype(int)
    score_eval = np.mean(np.sum(mat/np.log2(np.arange(2, mat.shape[1] + 2)),axis = 1))
    return 'ndcg5', score_eval


def score(probabilities, y, le):
    assert(probabilities.shape[0] == y.shape[0])
    count = probabilities.shape[0]
    scores = np.zeros(count)
    for i in range(count):
        truth = le.inverse_transform(y[i])
        predicted_i = le.inverse_transform(np.argsort(probabilities[i]))[::-1][:5]
        predicted_i_v = np.zeros(5, dtype=np.int32) + (predicted_i == truth)
        scores[i] = ndcg_at_k(predicted_i_v)
    return np.average(scores)


def print_xgboost_scores(classifier, columns=None):
    if hasattr(classifier, 'booster'):
        bst = classifier.booster()
        f_scores = bst.get_fscore()
        f_scores_sorted = sorted(f_scores.items(), key=operator.itemgetter(1))
        if columns is None:
            print('Features importance score: ', f_scores_sorted)
        else:
            print('Features importance score: ')
            indices = []
            for feature in f_scores_sorted:
                f_label, f_score = feature
                f_idx = int(f_label[1:])
                indices.append(f_idx)
                print(columns[f_idx] + ": " + str(f_score))
            print('Not used features: ')
            for idx, column in enumerate(columns):
                if idx not in indices:
                    print(column)
