#-------------------------------------------------------------------------
# AUTHOR: Anh Tu Nguyen
# FILENAME: naive_bayes.py
# SPECIFICATION: Performs Naïve Bayes classification on weather test data.
#               It reads training data from weather_training.csv and test data 
#               from weather_test.csv, and prints the classification and 
#               confidence for each test instance if confidence >= 0.75.
# FOR: CS 4210- Assignment #2, Task 5b
# TIME SPENT: 45 minutes
#-------------------------------------------------------------------------
# IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH
# AS numpy OR pandas. You have to work here only with standard dictionaries, lists, and arrays.

from sklearn.naive_bayes import GaussianNB
import csv

# ---------------------------
# Define mapping dictionaries for categorical features
# ---------------------------
outlook_map = {'Sunny': 1, 'Overcast': 2, 'Rain': 3}
temperature_map = {'Hot': 1, 'Mild': 2, 'Cool': 3}
humidity_map = {'High': 1, 'Normal': 2}
wind_map = {'Weak': 1, 'Strong': 2}
# For the class: Yes = 1, No = 2
play_map = {'Yes': 1, 'No': 2}
# Reverse mapping for printing predicted labels
play_reverse = {1: 'Yes', 2: 'No'}

# ---------------------------
# Reading the training data from weather_training.csv
# ---------------------------
training_data = []
with open('weather_training.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    header = next(reader)  # skip header
    for row in reader:
        training_data.append(row)

# ---------------------------
# Transform training features into a 4D array X and training labels into Y
# Format of each row in X: [Outlook, Temperature, Humidity, Wind]
# ---------------------------
X = []
Y = []
for row in training_data:
    # row: Day, Outlook, Temperature, Humidity, Wind, PlayTennis
    outlook = outlook_map.get(row[1], 0)
    temperature = temperature_map.get(row[2], 0)
    humidity = humidity_map.get(row[3], 0)
    wind = wind_map.get(row[4], 0)
    X.append([float(outlook), float(temperature), float(humidity), float(wind)])
    Y.append(play_map.get(row[5], 0))

# ---------------------------
# Fitting the Naïve Bayes classifier to the training data
# ---------------------------
clf = GaussianNB(var_smoothing=1e-9)
clf.fit(X, Y)

# ---------------------------
# Reading the test data from weather_test.csv
# ---------------------------
test_data = []
with open('weather_test.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    test_header = next(reader)  # skip header
    for row in reader:
        test_data.append(row)

# ---------------------------
# Printing the header for the solution
# ---------------------------
# Print the header with alignment
print("{:<6} {:<10} {:<11} {:<10} {:<7} {:<10} {:<10}".format(
        "Day", "Outlook", "Temperature", "Humidity", "Wind", "PlayTennis", "Confidence"
    ))

# ---------------------------
# For each test sample, predict the class and print if the confidence >= 0.75
# ---------------------------

for row in test_data:
    # row: Day, Outlook, Temperature, Humidity, Wind, PlayTennis (true label)
    day = row[0]
    outlook = outlook_map.get(row[1], 0)
    temperature = temperature_map.get(row[2], 0)
    humidity = humidity_map.get(row[3], 0)
    wind = wind_map.get(row[4], 0)
    
    test_sample = [float(outlook), float(temperature), float(humidity), float(wind)]
    probabilities = clf.predict_proba([test_sample])[0]
    max_prob = max(probabilities)
    predicted_class_num = clf.predict([test_sample])[0]
    predicted_label = play_reverse.get(predicted_class_num, "Unknown")
    


    # Only output if the maximum predicted probability (confidence) is >= 0.75
    if max_prob >= 0.75:
        # Print the test instance in the desired format
        print("{:<6} {:<10} {:<11} {:<10} {:<7} {:<10} {:<10}".format(
        day, 
        row[1], 
        row[2], 
        row[3], 
        row[4], 
        predicted_label, 
        f"{max_prob:.2f}"
    ))
