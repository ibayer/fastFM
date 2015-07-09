# Author: Immanuel Bayer
# License: BSD 3 clause


def kendall_tau(a, b):
    n_samples = a.shape[0]
    assert a.shape == b.shape
    n_concordant = 0
    n_disconcordant = 0

    for i in range(n_samples):
        for j in range(i+1, n_samples):
            if a[i] > a[j] and b[i] > b[j]:
                n_concordant = n_concordant + 1
            if a[i] < a[j] and b[i] < b[j]:
                n_concordant = n_concordant + 1

            if a[i] > a[j] and b[i] < b[j]:
                n_disconcordant = n_disconcordant + 1
            if a[i] < a[j] and b[i] > b[j]:
                n_disconcordant = n_disconcordant + 1
    return (n_concordant - n_disconcordant) / (.5 * n_samples *
                                               (n_samples - 1))
