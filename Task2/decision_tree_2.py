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

# Define mapping dictionaries for the categorical features and target
age_map = {'Young': 1, 'Prepresbyopic': 2, 'Presbyopic': 3}
spectacle_map = {'Myope': 1, 'Hypermetrope': 2}
astigmatism_map = {'Yes': 1, 'No': 2}
tear_map = {'Normal': 1, 'Reduced': 2}
lenses_map = {'Yes': 1, 'No': 2}  # target mapping

# List of training set filenames
dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']

# For each training set file:
for ds in dataSets:

    dbTraining = []  # raw training rows
    X = []  # transformed training features (4D vector)
    Y = []  # training target labels

    # Reading the training data from the CSV file
    with open(ds, 'r') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)  # skip header
        for i, row in enumerate(reader):
            dbTraining.append(row)

    # Transform the original categorical training features to numbers
    # Training data columns are assumed as:
    # Age, Spectacle, Astigmatism, Tear, Lenses
    for row in dbTraining:
        # Convert each categorical feature using the mapping dictionaries:
        age = age_map.get(row[0].strip(), 0)
        spectacle = spectacle_map.get(row[1].strip(), 0)
        astigmatism = astigmatism_map.get(row[2].strip(), 0)
        tear = tear_map.get(row[3].strip(), 0)
        X.append([float(age), float(spectacle), float(astigmatism), float(tear)])
        # Transform the class label (Lenses) to number
        Y.append(lenses_map.get(row[4].strip(), 0))

    # We'll repeat the training and testing process 10 times and accumulate the accuracy
    total_accuracy = 0.0

    for iteration in range(10):

        # Fitting the decision tree to the training data with max_depth=5
        clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5)
        clf = clf.fit(X, Y)

        # Reading the test data from 'contact_lens_test.csv'
        dbTest = []  # raw test rows
        with open('contact_lens_test.csv', 'r') as csvfile:
            reader = csv.reader(csvfile)
            test_header = next(reader)  # skip header
            for row in reader:
                dbTest.append(row)

        correct = 0
        total = 0

        # For each test instance, transform features and use the tree for prediction
        for data in dbTest:
            # Test data columns: Age, Spectacle, Astigmatism, Tear, Lenses (ground truth)
            age = age_map.get(data[0].strip(), 0)
            spectacle = spectacle_map.get(data[1].strip(), 0)
            astigmatism = astigmatism_map.get(data[2].strip(), 0)
            tear = tear_map.get(data[3].strip(), 0)
            test_features = [float(age), float(spectacle), float(astigmatism), float(tear)]
            # Predict the class using the decision tree; [0] to get an integer prediction
            class_predicted = clf.predict([test_features])[0]
            # Compare with the true label (Lenses) in data[4]
            true_label = lenses_map.get(data[4].strip(), 0)
            if class_predicted == true_label:
                correct += 1
            total += 1

        # Calculate the accuracy for this iteration
        iteration_accuracy = correct / total
        total_accuracy += iteration_accuracy

    # Calculate average accuracy over the 10 runs
    avg_accuracy = total_accuracy / 10

    print("Final accuracy when training on {}: {:.4f}".format(ds, avg_accuracy))
