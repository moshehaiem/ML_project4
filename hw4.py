import numpy as np
from sklearn.tree import DecisionTreeClassifier
from scipy import stats
import numbers


def load_data(filename):
    data = np.load(filename)
    data_x = data[:, :-1]
    data_y = data[:, -1].astype(int)
    print(f"x: {data_x.shape}, y:{data_y.shape}")
    return data_x, data_y


class Bagging(object):
    clf = []
    def __init__(self, n_classifiers, max_depth):
        '''
        Input:
            n_classifiers: number of trees in the ensemble. int
            max_depth: maximum depth allowed for every tree built. It should not exceed 20. int
        '''
        ###############################
        # TODO: your implementation
        ###############################
        self.n_classifiers = n_classifiers
        self.max_depth = max_depth
        for i in range(self.n_classifiers):
            self.clf.append(0)
        
    
    def train(self, X, y):
        '''
        Build an ensemble.
        Input:
            X: Feature vector of shape (N, D). N - number of training samples; D - number of features. np ndarray
            y: label vector of shape (N,). np ndarray
        '''
        for i in range(self.n_classifiers):
            self.clf[i] = DecisionTreeClassifier(max_depth = self.max_depth)
            randComb = np.random.choice(1000, 1000, replace=True)
            self.clf[i] = self.clf[i].fit(X[randComb], y[randComb])
        ###############################
        # TODO: your implementation
        ###############################
        
    
    def test(self, X):
        '''
        Predict labels X. 
        Input:
            X: testing feature vectors of shape (N, D). np array
        Output:
            prediction: label vector of shape (N,). np array, dtype=int
        '''
        pred = np.zeros((self.n_classifiers, X.shape[0]))
        counter1 = []
        counter0 = []

        for i in range(X.shape[0]):
            counter1.append(0)
            counter0.append(0)

        for i in range(self.n_classifiers):
            pred[i] = self.clf[i].predict(X)
            



        for i in range(self.n_classifiers):
            for j in range(X.shape[0]):
                if pred[i][j] == 0:
                    counter0[j] += 1
                elif pred[i][j] == 1:
                    counter1[j] += 1
        
        predCombined = np.zeros((X.shape[0]))
        for i in range(X.shape[0]):
            if counter1[i] > counter0[i]:
                predCombined[i] = 1
            else:
                predCombined[i] = 0
        return predCombined
        ###############################
        # TODO: your implementation
        ###############################
        
    
    
class Boosting(object):
    clf = []
    alphaArr = []
    def __init__(self, n_classifiers, max_depth):
        '''
        Input:
            n_classifiers: the maximum number of trees at which the boosting is terminated. int
            max_depth: maximum depth allowed for every tree built. It should not exceed 2. int
        '''
        if max_depth!=1 and max_depth!=2:
            raise ValueError('max_depth can only be 1 or 2!')
        self.max_depth = max_depth
        self.n_classifiers = n_classifiers
        for i in range(self.n_classifiers):
            self.clf.append(0)
            self.alphaArr.append(0)
        ###############################
        # TODO: your implementation
        ###############################
        
        
    def train(self, X, y):
        '''
        Train an adaboost.
        Input:
            X: Feature vector of shape (N, D). N - number of training samples; D - number of features. np ndarray
            y: label vector of shape (N,). np ndarray
        '''
        
        weights = np.ones(X.shape[0]) / X.shape[0]
        for t in range(self.n_classifiers):
            self.clf[t] = DecisionTreeClassifier(splitter='random', max_depth = self.max_depth)
            self.clf[t].fit(X, y, sample_weight = weights)
            prediction = self.clf[t].predict(X)

            error = 0
            size = prediction.shape[0]
            for i in range(size):
                error += weights[i] * (prediction[i] != y[i])
            error /= np.sum(weights)

            alpha = (np.log(((1-error)/error)**.5))
            self.alphaArr[t] = alpha
            for i in range(X.shape[0]):
                if prediction[i] != y[i]: 
                    weights[i] *= np.exp(alpha)
                else:
                    weights[i] *= np.exp(-alpha)
            
            
            weights/=np.sum(weights)
        
        
        self.alphaArr /= np.sum(self.alphaArr)

        ###############################
        # TODO: your implementation
        ###############################
        
    
    def test(self, X):
        '''
        Predict labels X. 
        Input:
            X: testing feature vectors of shape (N, D). np array
        Output:
            prediction: label vector of shape (N,). np array, dtype=int
        '''
        totalClassifiers = []
        addedClassifier = np.zeros((X.shape[0]))
        
        for i in range(self.n_classifiers):
            totalClassifiers.append(0)

        
        for i in range(self.n_classifiers):
            totalClassifiers[i] = self.alphaArr[i] * self.clf[i].predict(X)
        

        for i in range(self.n_classifiers):
            for j in range(X.shape[0]):
                addedClassifier[j] += totalClassifiers[i][j]
        

        

        for i in range(X.shape[0]):
            if addedClassifier[i] >= .5:
                addedClassifier[i] = 1
            else:
                addedClassifier[i] = 0

        

        return addedClassifier
        ###############################
        # TODO: your implementation
        ###############################
        
    
    
# Please do not modify the variable names
X_train, y_train = load_data('winequality-red-train-2class.npy')
X_test, y_test = load_data('winequality-red-test-2class.npy')
bagging = Bagging(50, 20)
boosting = Boosting(1000, 2)
bagging.train(X_train, y_train)
boosting.train(X_train, y_train)
print('Bagging test accuracy: ', np.mean(bagging.test(X_test)==y_test))
print('AdaBoost test accuracy: ', np.mean(boosting.test(X_test)==y_test))