import matplotlib.pyplot as plt
import numpy as np
import h5py
from sklearn.metrics import r2_score

class Evaluation(object):
    def __init__(self):
        pass

    @staticmethod
    def mae_(target, output):
        return np.mean(np.abs(target - output))

    @staticmethod
    def mape_(target, output):
        return np.mean(np.abs(target - output) / (target + 1))

    @staticmethod
    def rmse_(target, output):
        return np.sqrt(np.mean(np.power(target - output, 2)))

    @staticmethod
    def r2_(target, output):
        return r2_score(target, output)
    @staticmethod
    def total(target, output):
        mae = Evaluation.mae_(target, output)
        mape = Evaluation.mape_(target, output)
        rmse = Evaluation.rmse_(target, output)
        r2=Evaluation.r2_(target,output)
        return mae, mape, rmse,r2


