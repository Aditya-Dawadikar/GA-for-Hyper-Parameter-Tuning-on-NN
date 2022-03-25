import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix


base_path = 'data/'

class DataSource:
    def __init__(self):
        # train and test data
        self.train = pd.read_csv(base_path+'train.csv')
        self.test = pd.read_csv(base_path+'test.csv')
        
        # dependent and independent variables
        self.X_train = self.train.iloc[:,:11]
        self.X_test = self.test.iloc[:,:11]
        self.y_train = self.train.iloc[:,11]
        self.y_test = self.test.iloc[:,11]
        
        # scaling data before use
        scalar = StandardScaler()
        self.X_train = scalar.fit_transform(self.X_train)
        self.X_test = scalar.transform(self.X_test)
        pass
    
    def get_train_data(self):
        return self.X_train,self.y_train
    
    def get_test_data(self):
        return self.X_test,self.y_test
    
class NeuralNetwork:
    def __init__(self,layers,max_iterations,activation_function):
        # create model from given parameters
        # layers: list of integers representing the number of neurons in each layer
        # max_iterations: integer representing max number of iterations for training
        # activation_function: String keyword for available activation functions ['relu','tanh','logistic']
        # solver_function: String keyword for available solvers ['adam','lbfgs']
        self.classifier = MLPClassifier(
                                    hidden_layer_sizes=layers, 
                                    max_iter=max_iterations,
                                    activation=activation_function, 
                                    solver='adam', 
                                    early_stopping=True
                                )
        pass

    def train(self,X_train,y_train):
        # train model
        self.classifier.fit(X_train, y_train)
        pass

    def evaluate_model(self,X_test,y_test):
        # evaluate model and return accuracy score
        # accuracy score will be considered as the fitness score
        y_pred = self.classifier.predict(X_test)
        
        acc_scr = accuracy_score(y_test, y_pred)
        return acc_scr
