import numpy as np


def encoder(y_train,a,b):
    y_train.flatten()
    y_train[y_train==a]=0
    y_train[y_train==b]=1
    encoded=y_train.astype(np.int64)
    return encoded
class logisticregression:
    def __init__(self,n=50000,lr=0.0001):
        self.lr=lr
        self.n=n
        self.A=None
    def fit(self,x_train,y_train):
        new_row=np.ones((x_train.shape[0],1))
        biased_x_train=np.hstack([new_row,x_train])

        self.A=np.zeros((biased_x_train.shape[1],1))

        y_train = y_train.reshape(-1, 1)

        for i in range(self.n):
            z=np.dot(biased_x_train,self.A)
            a=(1/(1+np.exp(z)))

            x=a-y_train
            grad=((-1)/(biased_x_train.shape[0]))*(np.dot(biased_x_train.T,x))
            self.A-=self.lr*grad

    def predict(self,X):
        new_row=np.ones((X.shape[0],1))
        X_final=np.hstack([new_row,X])
        z=np.dot(X_final,self.A)
        pred=(1/(1+np.exp(z)))
        output=[]
        for i in range(len(pred)):
            val=pred[i][0]
            if val>=0.5:
                output.append(1)
            else:
                output.append(0)
        output_array=np.array(output)
        return output_array
    def test(self,X):
        new_row=np.ones((X.shape[0],1))
        X_final=np.hstack([new_row,X])
        z=np.dot(X_final,self.A)
        pred=(1/(1+np.exp(z)))
        return pred


def logloss(y_pred,y_true):
    m=len(y_true)
    y_true=y_true.reshape(-1, 1)
    loss=(np.sum((y_true*(np.log(y_pred)))+((1-y_true)*(np.log(1-y_pred)))))*((-1)/m)
    return loss

def Confusion_Matrix(y_test,y_pred):
    c=np.zeros((2,2),dtype=int)
    for i in range(len(y_test)):
        if y_test[i]==y_pred[i]==0:
            c[0][0]+=1
        if y_test[i]==y_pred[i]==1:
            c[1][1]+=1
        if y_test[i]==1 and y_pred[i]==0:
            c[0][1]+=1
        if y_test[i]==0 and y_pred[i]==1:
            c[1][0]+=1
    return c




