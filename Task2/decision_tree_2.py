#-------------------------------------------------------------------------
# AUTHOR: Anh Tu Nguyen
# FILENAME: decision_tree_2.py
# SPECIFICATION: Reads 3 contact lens training sets and a test set, builds
#                a decision tree (max_depth=5) for each training set,
#                repeats training and testing 10 times, and outputs the
#                average accuracy of each model.
# FOR: CS 4210- Assignment #2, task 2
# TIME SPENT: 45 minutes
#-------------------------------------------------------------------------
# IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH
# AS numpy OR pandas. You have to work here only with standard dictionaries, lists, and arrays

from sklearn import tree
import csv
import random

# Define mapping dictionaries for categorical features and target
# Assumed training/test data columns: Age, Spectacle, Astigmatism, Tear, Lenses
age_map = {'Young': 1, 'Prepresbyopic': 2, 'Presbyopic': 3}
spectacle_map = {'Myope': 1, 'Hypermetrope': 2}
astigmatism_map = {'Yes': 1, 'No': 2}
tear_map = {'Normal': 1, 'Reduced': 2}
lenses_map = {'Yes': 1, 'No': 2}  # target mapping

# List of training set filenames
dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']

# Process each training set
for ds in dataSets:
    
    dbTraining = []  # raw training data rows
    # Read the training data file
    with open(ds, 'r') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)  # skip header
        for row in reader:
            dbTraining.append(row)
    
    # Pre-transform all training data rows into numerical features (stored as list of lists)
    # Each row is assumed to be: Age, Spectacle, Astigmatism, Tear, Lenses
    allTrainingData = []  # each element is [age, spectacle, astigmatism, tear, lenses]
    for row in dbTraining:
        age = age_map.get(row[0].strip(), 0)
        spectacle = spectacle_map.get(row[1].strip(), 0)
        astigmatism = astigmatism_map.get(row[2].strip(), 0)
        tear = tear_map.get(row[3].strip(), 0)
        target = lenses_map.get(row[4].strip(), 0)
        allTrainingData.append([float(age), float(spectacle), float(astigmatism), float(tear), target])
    
    # We'll perform 10 iterations and average the accuracy
    total_accuracy = 0.0

    for iteration in range(10):
        # To introduce variability, we create a bootstrap sample from allTrainingData.
        training_sample = random.choices(allTrainingData, k=len(allTrainingData))
        
        X = []  # training features (list of 4-element lists)
        Y = []  # training labels (list of integers)
        for row in training_sample:
            # row: [age, spectacle, astigmatism, tear, lenses]
            X.append([row[0], row[1], row[2], row[3]])
            Y.append(row[4])
        
        # Train the decision tree classifier (using entropy and max_depth=5)
        clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5)
        clf = clf.fit(X, Y)
        
        # Read the test data from 'contact_lens_test.csv'
        dbTest = []
        with open('contact_lens_test.csv', 'r') as csvfile:
            reader = csv.reader(csvfile)
            test_header = next(reader)  # skip header
            for row in reader:
                dbTest.append(row)
        
        correct = 0
        total = 0
        # For each test instance, transform the features and predict the class
        for data in dbTest:
            # Test data columns: Age, Spectacle, Astigmatism, Tear, Lenses (ground truth)
            test_age = age_map.get(data[0].strip(), 0)
            test_spectacle = spectacle_map.get(data[1].strip(), 0)
            test_astigmatism = astigmatism_map.get(data[2].strip(), 0)
            test_tear = tear_map.get(data[3].strip(), 0)
            test_features = [float(test_age), float(test_spectacle), float(test_astigmatism), float(test_tear)]
            class_predicted = clf.predict([test_features])[0]
            true_label = lenses_map.get(data[4].strip(), 0)
            if class_predicted == true_label:
                correct += 1
            total += 1
        
        iteration_accuracy = correct / total
        total_accuracy += iteration_accuracy

    # Average accuracy over 10 iterations
    avg_accuracy = total_accuracy / 10.0
    print("Final accuracy when training on {}: {:.4f}".format(ds, avg_accuracy))