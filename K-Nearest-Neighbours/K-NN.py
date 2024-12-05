
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Distance function euclidean
def distance(x,y):
    return (np.sum((np.array(x)-np.array(y))**2))**(1/2)

# KNN main algorithm
def predict_quadrant(a,training,t_arrays,k=11):

        distances=[]
        for data in t_arrays:
            dist=distance(a,data)
            distances.append(dist)

        initial_output=np.array(distances)
        sorted_distances=np.argsort(initial_output)
        knn=sorted_distances[:k]

        knn_map=[]

        for idx in knn:
            y=training.iloc[idx]
            knn_map.append(y)

        Quadrants_array=np.array(knn_map)
        Quadrants=Quadrants_array[:,2].tolist()

        q1=Quadrants.count("q1")
        q2=Quadrants.count("q2")
        q3=Quadrants.count("q3")
        q4=Quadrants.count("q4")

        quadrant_counts = {"q1": q1, "q2": q2, "q3": q3, "q4": q4}

        # Find the quadrant with the maximum count
        def get_min_distance(quadrant):
            # Get the distances for the specific quadrant's nearest neighbors
            relevant_distances = [distances[idx] for idx in knn if training.iloc[idx]["quadrant"] == quadrant]

            # If there are relevant distances, return the minimum, otherwise return a large value (so it doesn't win)
            return min(relevant_distances) if relevant_distances else float('inf')

        max_quadrant = max(quadrant_counts, key=lambda x: (quadrant_counts[x], -get_min_distance(x)))
        return max_quadrant

# Loading training data
training=pd.read_csv("training.csv")
t_arrays=[np.array(row) for row in training[["x","y"]].values]

# Loading test data
test=pd.read_csv("test.csv")
test_arrays=[np.array(row) for row in test[["x","y"]].values]
test_quad=test["quadrant"].tolist()

# Test
score=0
for i in range(len(test_arrays)):
    a=test_arrays[i]
    predicted_quadrant=predict_quadrant(a,training,t_arrays,11)
    if predicted_quadrant==test_quad[i]:
        score+=1


percentage=(score*100)/len(test_quad)
print(f"Percentage correctness={percentage}%")

# Combining user and test data
combined_data = pd.concat([training, test], ignore_index=True)
combined_arrays = [np.array(row) for row in combined_data[["x", "y"]].values]

import matplotlib.pyplot as plt

# Extract data for each quadrant
q2_data = combined_data[combined_data["quadrant"] == "q2"]
q3_data = combined_data[combined_data["quadrant"] == "q3"]
q4_data = combined_data[combined_data["quadrant"] == "q4"]
q1_data = combined_data[combined_data["quadrant"] == "q1"]

# Plot each quadrant
plt.figure(figsize=(8, 8))

plt.scatter(q1_data["x"], q1_data["y"], color="red", label="Q1", alpha=0.7)
plt.scatter(q2_data["x"], q2_data["y"], color="blue", label="Q2", alpha=0.7)
plt.scatter(q3_data["x"], q3_data["y"], color="green", label="Q3", alpha=0.7)
plt.scatter(q4_data["x"], q4_data["y"], color="purple", label="Q4", alpha=0.7)

# Add labels and legend
plt.title("Training Data by Quadrants")
plt.xlabel("X")
plt.ylabel("Y")
plt.axhline(0, color="black", linestyle="--", linewidth=0.7)
plt.axvline(0, color="black", linestyle="--", linewidth=0.7)
plt.legend()
plt.grid(True)
plt.savefig("training_data_plot.png", dpi=300, bbox_inches='tight')


# User
user_inputx=float(input("x: "))
user_inputy=float(input("y: "))

input_list=[user_inputx,user_inputy]

user_quad=predict_quadrant(input_list,combined_data,combined_arrays)

print(f"predicted quadrant is {user_quad}")