import numpy as np


# train
apriory_t = {'NDF': 0.583461, 'US': 0.292225, 'other': 0.047291, 'FR': 0.023531, 'CA': 0.006690, 'GB': 0.010892, 'ES': 0.010536, 'IT': 0.013285, 'PT': 0.001021, 'NL': 0.003574, 'DE': 0.004970, 'AU': 0.002525 }

# train with session
apriory_t_s = {'NDF': 0.610188, 'US': 0.272235, 'other': 0.049516, 'FR': 0.019440, 'CA': 0.005961, 'GB': 0.009903, 'ES': 0.009578, 'IT': 0.013263, 'PT': 0.001124, 'NL': 0.003346, 'DE': 0.003387, 'AU': 0.002059 }

# test calculated
apriory_c = {'NDF': 0.6635,  'US': 0.2307,  'other': 0.0380,  'FR': 0.0175,  'CA': 0.0066,  'GB': 0.0092,  'ES': 0.00725,  'IT': 0.0130,  'PT': 0.0015,  'NL': 0.0043,  'DE': 0.0039,  'AU': 0.0026 }

# test
apriory = {'NDF': 0.67909,  'US': 0.23470,  'other': 0.03403,  'FR': 0.01283,  'CA': 0.00730,  'GB': 0.00730,  'ES': 0.00725,  'IT': 0.01004,  'PT': 0.00075,  'NL': 0.00215,  'DE': 0.00344,  'AU': 0.00113 }


def print_probabilities(probabilities):
    print('probabilities means: ',  ' '.join('{0:.4f}'.format(p) for p in list(probabilities.mean(axis=0))))
    print('probabilities sum of means: ', np.sum(probabilities.mean(axis=0).sum()))


def correct_probs(probs, le):
    probs = np.copy(probs)
    print()
    labels=le.inverse_transform(range(probs.shape[1]))
    ratios = [apriory[l] / apriory_c[l] for l in labels]
    for i in range(probs.shape[0]):
        probs[i] = correct_prob(probs[i], le, labels, ratios)
    return probs


def correct_prob(prob, le, labels, ratios):
    n = len(labels)

    denominator = 0
    for j in range(n):
        denominator += ratios[j] * prob[j]

    for i in range(n):
        prob[i] = ratios[i] * prob[i] / denominator

    return prob


def adjust_test_data(x_test, y_test, le):
    ndf_label = le.transform(['NDF'])[0]

    n = x_test.shape[0]
    assert(y_test.shape[0] == n)

    perm = np.random.permutation(n)
    x_test = x_test[perm,:]
    y_test = y_test[perm]

    ndf_ind = np.where(y_test == ndf_label)[0]
    not_ndf_ind = np.where(y_test != ndf_label)[0]

    not_ndf_count = int(0.657 * not_ndf_ind.shape[0])
    not_ndf_ind = not_ndf_ind[:not_ndf_count]

    ind = np.concatenate([ndf_ind, not_ndf_ind])
    x_test = x_test[ind]
    y_test = y_test[ind]

    return x_test, y_test

