@vectorize(nopython=True, cache=False)
def _PPV(a, b):
    if a > b:
        return 1
    return 0

@njit(fastmath=True, parallel=True, cache=False)
def _static_transform_uni(X, parameters, indices):
    """Transform a 2D collection of univariate time series.

    Implemented separately to the multivariate version for numba efficiency reasons.
    See issue #1778.
    """
    n_cases, n_timepoints = X.shape
    _, _, dilations, n_features_per_dilation, biases = parameters
    n_kernels = len(indices)
    n_dilations = len(dilations)
    f = n_kernels * np.sum(n_features_per_dilation)
    features = np.zeros((n_cases, f), dtype=np.float32)
    for i in prange(n_cases):
        _X = X[i]
        A = -_X
        G = 3 * _X
        f_start = 0
        for j in range(n_dilations):
            _padding0 = j % 2
            dilation = dilations[j]
            padding = 8 * dilation // 2
            n_features = n_features_per_dilation[j]
            C_alpha = np.zeros(n_timepoints, dtype=np.float32)
            C_alpha[:] = A
            C_gamma = np.zeros((9, n_timepoints), dtype=np.float32)
            C_gamma[4] = G
            start = dilation
            end = n_timepoints - padding
            for gamma_index in range(4):
                C_alpha[-end:] = C_alpha[-end:] + A[:end]
                C_gamma[gamma_index, -end:] = G[:end]
                end += dilation
            for gamma_index in range(5, 9):
                C_alpha[:-start] = C_alpha[:-start] + A[start:]
                C_gamma[gamma_index, :-start] = G[start:]
                start += dilation
            for k in range(n_kernels):
                f_end = f_start + n_features
                _padding1 = (_padding0 + k) % 2
                a, b, c = indices[k]
                C = C_alpha + C_gamma[a] + C_gamma[b] + C_gamma[c]
                if _padding1 == 0:
                    for f in range(n_features):
                        features[i, f_start + f] = _PPV(C, biases[f_start + f]).mean()
                else:
                    for f in range(n_features):
                        features[i, f_start + f] = _PPV(C[padding:-padding], biases[f_start + f]).mean()
                f_start = f_end
    return features