import numpy as np
    
class Node:
    def __init__(self,feature=None,threshold=None,right=None,left=None,*,value=None):
        self.feature=feature
        self.threshold=threshold
        self.left=left
        self.right=right
        self.value=value
    
    def is_leaf(self):
        return self.value is not None



class DecisionTree:

    def __init__(self,min_samples_split=2,max_depth=100):
        self.max_depth=max_depth
        self.min_samples_split=min_samples_split
        self.root=None

    def entropy(self, y):
        y=y.flatten()
        proportions = np.bincount(y) / len(y)
        val = -np.sum([p * np.log2(p) for p in proportions if p > 0])
        return val

    def split(self,X,thresh):
        positive=np.argwhere(X<=thresh).T[0]
        negetive=np.argwhere(X>thresh).T[0]
        return positive,negetive

    def information_gain(self,X,y,thresh):
        e=self.entropy(y)
        left,right=self.split(X,thresh)
        if len(left)==0 or len(right)==0:
            return 0,None,None
        else:
            y1=y[left]
            y2=y[right]
            e1=(y1.shape[0]/y.shape[0])*self.entropy(y1)
            e2=(y2.shape[0]/y.shape[0])*self.entropy(y2)
            ig=e-(e1+e2)
            return ig,left,right
        
    def best_split(self,X,y,features):
        values={'score':0,'feat':None,'thresh':None,'left':None,'right':None}
        for feat in features:
            X_feat=X[:,feat]
            thresholds=np.unique(X_feat)
            for thresh in thresholds:
                score,left_idx,right_idx=self.information_gain(X_feat,y,thresh)

                if score>values['score']:
                    values['score']=score
                    values['feat']=feat
                    values['thresh']=thresh
                    values['left']=left_idx
                    values['right']=right_idx
        return values['feat'],values['thresh'],values['left'],values['right']
    
    def tree(self,X,y,depth=0):
        n_val,n_features=X.shape
        classes=np.unique(y)
        features=range(n_features)
        if n_val<self.min_samples_split or depth==self.max_depth:
            y=y.flatten()
            leaf_val=np.bincount(y).argmax()
            return Node(value=leaf_val)
        
        elif len(classes)==1:
            return Node(value=classes[0])
        
        else:
            feat,thresh,left_idx,right_idx=self.best_split(X,y,features)
            if left_idx is None or right_idx is None:
                y=y.flatten()
                leaf_val=np.bincount(y).argmax()
                return Node(value=leaf_val)
            else: 
                left_node=self.tree(X[left_idx],y[left_idx],depth+1)
                right_node=self.tree(X[right_idx],y[right_idx],depth+1)
                return Node(feat,thresh,right_node,left_node)

    def fit(self, X_train, y_train):
        y_train=y_train.reshape(-1,1)
        self.root=self.tree(X_train,y_train)

    def predict_one(self,X,node):
        if node.is_leaf():
            return node.value
        if X[:,node.feature]<=node.threshold:
            return self.predict_one(X,node.left)
        else:
            return self.predict_one(X,node.right)

    def predict(self, X):
        predictions=[]
        for x in X:
            x=x.reshape(1,-1)
            predictions.append(self.predict_one(x,self.root))
        return predictions
    
    def accuracy(self,y_true,y_pred):
        total=len(y_true)
        right_values=0
        for i in range(total):
            true=y_true[i]
            pred=y_pred[i]
            if true==pred:
                right_values+=1
        return (right_values*100)/total

    def evaluate(self,X,y):
        y_pred=self.predict(X)
        y_true=y.flatten().tolist()
        return self.accuracy(y_true,y_pred)