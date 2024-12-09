import numpy as np

def exp_gradient(x,y_true,A,c):
    m=x.shape[0]
    y_pred=np.exp(np.dot(x,A))+c
    error=np.subtract(y_pred,y_true)
    dA=(1/m)*np.dot(x.T,(np.exp(np.dot(x,A))*error))
    dc=(1/m)*np.sum(error)
    return dA,dc

def exp_loss(x,y_true,A,c):
    m=x.shape[0]
    y_pred=np.exp(np.dot(x,A))+c
    mse=(1/m)*np.sum((y_pred-y_true)**2)
    return mse

class ExpRegression:
    def __init__(self,iter=5000,l_rate=0.001):
        self.iter=iter
        self.l_rate=l_rate
        self.A=None
        self.c=None
        self.train_loss=None


    def fit(self,x_train,y_train):
        m=x_train.shape[0]
        new_row=np.ones((x_train.shape[0],1))
        x_train=np.hstack([new_row,x_train])
        y_train=y_train.reshape(m,1)

        self.A=np.zeros((x_train.shape[1],1))
        self.c=0

        for i in range(self.iter):
            dA,dc= exp_gradient(x_train,y_train,self.A,self.c)
            self.A-=dA*self.l_rate
            self.c-=dc*self.l_rate
        self.train_loss=exp_loss(x_train,y_train,self.A,self.c)
        print('Final Loss:',self.train_loss)


    def evaluate(self,x_test,y_test):
        new_row=np.ones((x_test.shape[0],1))
        x_test=np.hstack([new_row,x_test])
        y_test=y_test.reshape(-1,1)
        mse=exp_loss(x_test,y_test,self.A,self.c)
        return mse

    def predict(self,X):
        new_row=np.ones((X.shape[0],1))
        X=np.hstack([new_row,X])
        y_pred=np.exp(np.dot(X,self.A))+self.c
        return y_pred