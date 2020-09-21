'''
Solving Linear Regression With Gradient Descent
Loss Function used: Sum of Squared Residuals (expected - predicted)^2
'''
from sklearn.model_selection import train_test_split
import numpy as np

class LinearRegression:
    # data set X = (n_samples,n_features),
    def __init__(self,X,Y, epochs = 10000,learning_rate = 0.001 ):
        self.n_samples = len(X)
        self.n_features = X.shape[1]
        self.parameters = np.zeros(self.n_features)
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.fit(X,Y)

    def fit(self,X,Y):
        for _ in range(self.epochs):#number of iterations
            new_parameters = self.parameters.copy()
            print(self.total_error(X,Y))
            for j in range(len(self.parameters)):#modify each parameter to minimize the loss function
                dj = np.array([self.loss_function_prime(x,y,j) for x,y in zip(X,Y)]).sum() #derivate of the j parameter
                new_parameters[j] = new_parameters[j] - self.learning_rate * dj # update parameter with Gradient Descent
            self.parameters = new_parameters.copy()

    #prediction of a simple sample
    def predict(self,x):
        W = np.array(self.parameters).T
        X = x.T
        return W.T.dot(X)

    #error of a simple sample
    def loss_function(self,x,y):
        return (y - self.predict(x))**2

    #derivate of the loss function for the parameter j
    def loss_function_prime(self,x,y,j):
        return -2 * x[j] * (y - self.predict(x))

    def total_error(self,X,Y):
        return np.array([self.loss_function(x,y) for x,y in zip(X,Y)]).sum()

if __name__ == "__main__":

    # X = np.array([[1,0.5],[1,2.3],[1,2.9]])
    # Y = np.array([1.4,1.9,3.2])
    # ## IDEAL (0.95,0.64)
    X = np.array([[1,2],[1,6],[1,5],[1,7]])
    Y = np.array([3,10,4,13])

    linearRegressionModel = LinearRegression(X,Y)
    print(linearRegressionModel.parameters)
