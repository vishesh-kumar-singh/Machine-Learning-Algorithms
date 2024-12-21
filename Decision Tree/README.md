# Desicion Tree Algorithm from Scratch

This is my implementation of Decision Tree, where I built the whole algorithm from scratch. The aim was to predict the survival for the infamous Titanic Challenge. I tried both my own built decision tree and the one present in sckit-learn Library with same parameters to compare if my model is as good as the one in sckit-learn.(Spoler Alert: I don't know why I always performed better)

## Features of Program
1. For training algorithm accepts numpy array input in the sequence of x_train and y_train.
2. After training the algorithm can provide an output as predicted list of classes.
3. The model contains evaluation function which can provide the accuracy of tree.


## Results
The following were the accuracy results over Kaggle:

| Submission No.    | Model 1   | Model 2   |
|--------------------|-----------|-----------|
| S1 (Original Data) | 0.71052   | 0.75358   |
| S2 (Oversampling)  | 0.73444   | 0.74162   |
| S3 (Undersampling) | 0.71531   | 0.72099   |
| S4 (Original Data) | 0.73923   | 0.75199   |

Although over the test data which I made from the training data both model gave randomly one bwing high and other low or many times being simmilar, but on testing over the entire test data for submission, I don't know why for the same max_depth and min_samples_splits my model is always performing better than in sckit-learn library.  
Maybe I did better job than those who wrote the algorithm over sckit-learn :)



If you feel I am dumb enough to not understand the reason kindly reach me out at [LinkedIn](https://linkedin.com/in/thevishesh16) or [Instagram](https://instagram.com/vishesh_kumar_singh).



## Usage

Here's an example of how to use the decision tree implementation:

```python
import numpy as np
from decision_tree import DecisionTree

# Example data
x_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
y_train = np.array([0, 1, 0, 1])

# Initialize and train the model
model = DecisionTree(max_depth=3, min_samples_split=2)
model.fit(x_train, y_train)

# Predict
x_test = np.array([[1, 2], [3, 4]])
predictions = model.predict(x_test)
print(predictions)

# Evaluate
accuracy = model.evaluate(x_test, np.array([0, 1]))
print(f"Accuracy: {accuracy}")
```


