📄 1. Project Title: Predicting Online Learning Completion Using Machine Learning.

📌 Submitted by: Ashish Gupta
B.Tech CSE (AI), KIET Institute of Technology
📅 Date: 22 April 2025

📘 2. Introduction
In today's fast-paced digital education landscape, understanding learner behavior is crucial for improving course engagement and completion rates. This project aims to predict whether a student will complete an online course based on activity logs such as videos watched, assignments submitted, and forum interactions. By using machine learning classification models, institutions can identify students who may need additional support.


🧪 3. Methodology
We utilized a classification approach for this supervised learning problem. The steps followed in this project were:
- Data Collection: Retrieved learner activity logs from a CSV file.
- Preprocessing: Converted categorical data into numerical format and handled missing values.
- Model Selection: Used Random Forest classifier to predict course completion.
- Evaluation: Measured model performance using accuracy, precision, recall, and confusion matrix visualization.

💻 4. Code
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

# Load dataset
df = pd.read_csv("online_learning.csv")

# Convert categorical labels ('yes', 'no') to numerical (1, 0)
df["completed"] = df["completed"].map({"yes": 1, "no": 0})

# Split data into features and target
X = df.drop(columns=["completed"])
y = df["completed"]

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# Display results
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")

# Confusion matrix heatmap
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Not Completed", "Completed"], yticklabels=["Not Completed", "Completed"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix Heatmap")
plt.show()


📊 5. Output
The model successfully classified learners into two categories:
- Completed: Students who finish the course.
- Not Completed: Students who drop out.
The output was visualized using a confusion matrix to assess classification performance.

🙌 6. Credits
📌 Project By: Ashish Gupta
🛠 Tools Used: Python, Pandas, Scikit-learn, Matplotlib
📂 Data Source: online_learning.csv


