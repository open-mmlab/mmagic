import numpy as np


def sad(alpha, trimap, pred_alpha):
    assert (pred_alpha[trimap == 0] == 0).all()
    assert (pred_alpha[trimap == 255] == 1).all()
    sad = np.abs(pred_alpha - alpha).sum() / 1000
    return sad


def mse(alpha, trimap, pred_alpha):
    assert (pred_alpha[trimap == 0] == 0).all()
    assert (pred_alpha[trimap == 255] == 1).all()
    weight_sum = (trimap == 128).sum()
    if weight_sum != 0:
        mse = ((pred_alpha - alpha)**2).sum() / weight_sum
    else:
        mse = 0
    return mse
