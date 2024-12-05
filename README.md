# K-Nearest Neighbors (KNN) Algorithm Project

This project implements a K-Nearest Neighbors (KNN) algorithm from scratch in Python to classify data points into quadrants based on their `(x, y)` coordinates.

---

## Features
- Custom distance calculation using Euclidean distance.
- Prediction of quadrants (`Q1`, `Q2`, `Q3`, `Q4`) based on labeled training data.
- Cross-validation and accuracy testing.
- Visualization of training and combined data using Matplotlib.
- Dynamic selection of hyperparameter `k` (number of neighbors).
- Supports user input for quadrant prediction.

---

## Project Structure
- **`training.csv`**: Contains the training data with `(x, y)` coordinates and their corresponding quadrants.
- **`test.csv`**: Contains the test data for validating the model's accuracy.
- **`main.py`**: The main Python script implementing the KNN algorithm.
- **`README.md`**: Project documentation.
- **`visualization.png`**: Image showing the visualization of the training data split into quadrants.

---

## Prerequisites
- Python 3.8 or higher
- Libraries: `numpy`, `pandas`, `matplotlib`

Install the required libraries using:
```bash
pip install numpy pandas matplotlib
```

## Usage
Clone the repository:
```bash
git clone https://github.com/vishesh-kumar-singh/KNN-Algorithm.git
```
```bash
cd knn-quadrant-classifier
```

## Run the script:

```bash
python K-NN.py
```

## Predict the quadrant for a custom point:

Input the coordinates when prompted.
The script will return the predicted quadrant based on the combined training data.

## Visualize training data:

The main.py script will generate a plot saved as visualization.png in the project directory.

## Example
Input
```bash
x: -2.5
y: -1.4
```
Output
```bash
Predicted Quadrant: q4
```
## Results
Accuracy: The model achieves approximately 95.09803921568627% accuracy on the test dataset with k=11.
