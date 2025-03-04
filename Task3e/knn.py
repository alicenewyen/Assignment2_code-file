#-------------------------------------------------------------------------
# AUTHOR: Anh Tu Nguyen
# FILENAME: knn.py
# SPECIFICATION: Performs leave-one-out cross-validation on email classification using 1NN.
# FOR: CS 4210- Assignment #2
# TIME SPENT: 45 minutes
#-------------------------------------------------------------------------
# IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH
# AS numpy OR pandas. You have to work here only with standard vectors and arrays.

from sklearn.neighbors import KNeighborsClassifier
import csv


# Read the email classification data from CSV file
db = []
with open('email_classification.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0:  # skip header
            db.append(row)

# Initialize counters for error tracking
error_count = 0
total_instances = len(db)

# Perform Leave-One-Out Cross Validation (LOO-CV)
for i in range(total_instances):
    X = []  # training features
    Y = []  # training labels

    # Use each instance as the test sample in turn
    testSample = db[i]
    # Convert features of test sample (all columns except last) to float
    test_features = []
    for k in range(len(testSample) - 1):
        test_features.append(float(testSample[k]))
    true_label = testSample[-1]  # test sample's true class

    # Build training set: all instances except the test instance
    for j in range(total_instances):
        if j == i:
            continue  # skip the test sample
        row = db[j]
        # Convert features to float
        features = []
        for k in range(len(row) - 1):
            features.append(float(row[k]))
        X.append(features)
        # Append the label (as a string)
        Y.append(row[-1])

    # Create and fit the 1NN classifier using Euclidean distance (p=2)
    clf = KNeighborsClassifier(n_neighbors=1, p=2)
    clf = clf.fit(X, Y)

    # Predict the class for the test sample
    class_predicted = clf.predict([test_features])[0]

    # Compare predicted class to the true class
    if class_predicted != true_label:
        error_count += 1

# Calculate accuracy and error rate
accuracy = 1 - (error_count / total_instances)
error_rate = error_count / total_instances

# Print the results
print("\nLOO-CV Accuracy for 1NN: {:.4f}".format(accuracy))
print("LOO-CV Error Rate for 1NN: {:.4f}\n".format(error_rate))
