import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
import collections

class KNeighborsClassifier:
    def __init__(self, num_neighbors=3):
        self.num_neighbors = num_neighbors

    def fit(self, X_train, y_train):
        self.samples = (pd.concat([X_train, y_train], axis=1)).reset_index()

    def predict(self, X_test):
        X_test = X_test.reset_index()
        preds = []
        for i in range(len(X_test)):
            # Calculate distance
            #   between test sample and every existing sample
            distances = []
            for j in range(len(self.samples)):
                sample = (self.samples.drop(['index',0],axis=1)).iloc[j].tolist()
                test = (X_test.drop(['index'],axis=1)).iloc[i].tolist()
                distances.append(euclidean(sample, test))
            
            # Choose indices of the closest sample
            #   and Extract labels of the closest sample
            labels = []
            for k in range(self.num_neighbors):
                min_index = np.argmin(distances)
                del distances[min_index]
                labels.append(self.samples.loc[min_index, 0])

            # Count each labels and Classify test sample
            c = collections.Counter(labels)
            preds.append(c.most_common()[0][0])
        return preds