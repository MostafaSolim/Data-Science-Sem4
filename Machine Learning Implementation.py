from main import df, np, pd, plt, sns, PCA, StandardScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, classification_report

import os

os.environ["LOKY_MAX_CPU_COUNT"] = "4"
# a) Categorical Encoding: If your dataset contains categorical (non-numeric) features, onvert them
# into numerical format using techniques such as LabelEncoder or OneHotEncoder.
categorical_column = df['Department']

# One-Hot Encoding
one_hot_encoder = OneHotEncoder(sparse_output=False)
one_hot_encoded = one_hot_encoder.fit_transform(categorical_column.values.reshape(-1, 1))

# Add one-hot encoded columns to the DataFrame
one_hot_encoded_df = pd.DataFrame(one_hot_encoded, columns=one_hot_encoder.get_feature_names_out(['Department']))
df = pd.concat([df, one_hot_encoded_df], axis=1)

# Display the updated DataFrame
print(df.head())

# MACHINE LEARNING ALGORITHMS

data = df.drop('Employee_ID', axis=1)

# Encode categorical variables
categorical_cols = ['Gender', 'Marital_Status', 'Department', 'Job_Role', 'Overtime', 'Attrition']
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Separate features and target
X = data.drop('Attrition', axis=1)
y = data['Attrition']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# K-Nearest Neighbors (KNN):
# K-Nearest Neighbors (KNN):
# K-Nearest Neighbors (KNN):
# K-Nearest Neighbors (KNN):
# K-Nearest Neighbors (KNN):
# Credits to my colleague for making the code for the KNN

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)

# Evaluation
knn_accuracy = accuracy_score(y_test, y_pred_knn)
knn_confusion = confusion_matrix(y_test, y_pred_knn)
knn_precision = precision_score(y_test, y_pred_knn)
knn_recall = recall_score(y_test, y_pred_knn)

print('')
print('===============================================')
print('                   KNN')
print('===============================================')
print("Accuracy:", knn_accuracy)
print("Confusion Matrix:\n", knn_confusion)
print("Precision:", knn_precision)
print("Recall:", knn_recall)

# Tree
# Tree
# Tree
# Tree
# Tree
# Tree
# Tree

tree = DecisionTreeClassifier(random_state=42)

# Train the model
tree.fit(X_train_scaled, y_train)

# Make predictions
y_pred_tree = tree.predict(X_test_scaled)

# Evaluate the model
tree_accuracy = accuracy_score(y_test, y_pred_tree)
tree_confusion = confusion_matrix(y_test, y_pred_tree)
tree_precision = precision_score(y_test, y_pred_tree)
tree_recall = recall_score(y_test, y_pred_tree)

# Print evaluation metrics
print('')
print('===============================================')
print('                   Tree')
print('===============================================')
print("Decision Tree Accuracy:", tree_accuracy)
print("Decision Tree Confusion Matrix:\n", tree_confusion)
print("Decision Tree Precision:", tree_precision)
print("Decision Tree Recall:", tree_recall)
print("\nClassification Report:")
print(classification_report(y_test, y_pred_tree))

    #c) Training and Testing Sets: Split the features and target into training and testing sets.

#2. Apply Machine Learning Algorithms:
    #a) K-Nearest Neighbors (KNN):
    # • Train a KNN model using the training set.

    #b) Naive Bayes:
    #• Train a Naive Bayes classifier using the training set.

    #c) Additional Model:
    #• Choose and train two additional suitable machine learning models (e.g., Decision Tree, Support
    #Vector Machine, or Random Forest).

    #d) Bouns:
    #• Apply a deep learning model, if possible, for further comparison.

    #3. Model evaluation:
    #For each model, compute the following evaluation metrics in the test set.
        #• Accuracy: Overall percentage of correctly predicted instances.

        #• Confusion Matrix: A table that visualizes true vs. predicted classes.

        #• Recall: The model’s ability to capture all relevant cases (i.e., true positives).

        #• Precision: The quality of the positive predictions made by the model.

        #Comparison: Compare the models based on these metrics and decide which algorithm performs best.
        #Provide a clear reasoning behind your choice, considering factors such as data distribution,
        # model assumptions, and performance metrics.