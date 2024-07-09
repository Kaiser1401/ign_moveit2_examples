from river.linear_model import LogisticRegression, BayesianLinearRegression
from river import naive_bayes
from river import preprocessing
from river import metrics

import numpy as np

class Classifyer(object):
    def __init__(self):
        #self.model = LogisticRegression()
        self.model = naive_bayes.GaussianNB()
        self.scaler = preprocessing.StandardScaler()
        self._list_pred =[]
        self._list_outcome= []
        self.confusion = metrics.ConfusionMatrix()


    def scale(self,data):
        # my prescale
        # sqrt -> var to sigma
        da = np.array(data)
        ds = np.sqrt(da)

        #scaler
        ddata = self.data2dict(ds)
        self.scaler.learn_one(ddata)
        sdata = self.scaler.transform_one(ddata)
        return sdata



    def data2dict(self,data):
        d = {}
        i = 0
        for e in data:
            d[str(i)] = e
            i += 1
        return d

    def learn(self, data:list, label:bool):
        sdata = self.scale(data)
        self.model.learn_one(sdata, label)

    def predict(self, data:list):
        sdata = self.scale(data)
        pred = self.model.predict_one(sdata)
        pred_prob = self.model.predict_proba_one(sdata)
        print(pred_prob)
        return pred

    def storeOutcome(self, prediciton:bool, real:bool):
        if prediciton is None:
            return
        self._list_outcome.append(real)
        self._list_pred.append(prediciton)
        self.confusion.update(real,prediciton)

    def resetConfusion(self):
        self.confusion = metrics.ConfusionMatrix()

