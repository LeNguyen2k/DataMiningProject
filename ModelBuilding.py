import pandas as pd
import numpy as np
from pandas.core.reshape.pivot import crosstab
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import pydotplus

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Attributes that are changed to work with data:
# Ethnicity:
# Group A -> 0
# Group B -> 1
# Group C -> 2
# Group D -> 3
# Group E -> 4
#
# Parental Level of Education:
# Divided into 2 level for simplicity:
# +) Higher Education:
#       Associate's degree, 
#       College, 
#       Master's degree, 
#       Bachelor's degree
# +) High school
#
# Test preparation course:
# Dummified to TestPrep_none to simplify the data
# 0 -> completed
# 1 -> none
# 
# Lunch:
# Dummified to Lunch_standard to simplify the data
# 0 -> free/reduced
# 1 -> standard
#
# Math tier, Reading tier and Writing tier:
# All are dummified into MathTier_pass, ReadTier_pass, WriteTier_pass respectively:
# 0 -> fail
# 1 -> pass


def cleanData(dataSet):
    attribute_To_Drop = ["math score", "reading score", "writing score", "Extra math study time", "Extra reading study time", "Extra writing study time", 
                        "math_guess_min", "math_guess_max", "reading_guess_min", "reading_guess_max", 
                        "writing_guess_min", "writing_guess_max"]

    data = dataSet.drop(attribute_To_Drop, 1)


    data["parental level of education"] = data["parental level of education"].replace(["bachelor's degree", "college", "master's degree", "associate's degree"], "higher education")

    print("DataSet:\n", data)

    crossTab = pd.crosstab([data["Ethnicity"], data["parental level of education"], data["lunch"], data["test preparation course"]], data["Total tier"])
    print(crossTab)

    newSet = pd.get_dummies(data, columns=["Math tier", "Reading tier", "Writing tier", "parental level of education", "lunch", "test preparation course"], 
                            prefix=["MathTier", "ReadTier", "WriteTier", "ParentEdu", "Lunch", "TestPrep"], drop_first=True)

    temp = {"group A": 0, "group B": 1, "group C": 2, "group D": 3, "group E": 4}
    
    newSet["Ethnicity"] = newSet["Ethnicity"].map(temp)
    # newSet = newSet.drop(["Math tier", "Reading tier", "Writing tier"], 1)
    
    print("CLEANED DATASET:\n", newSet)

    return newSet

def splitDataSet(dataSet):
    train, test = train_test_split(newSet, test_size=0.2, shuffle=False)

    return train, test

def doDecisionTree(trainSet, testSet):
    Y = trainSet["Total tier"]
    X = trainSet.drop("Total tier", axis=1)
    
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X, Y)

    dot_data = tree.export_graphviz(clf, feature_names=X.columns, class_names=["Fail", "Pass"], filled=True, out_file=None)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_png("StudentTreeTrain.png")

    img = mpimg.imread('StudentTreeTrain.png')
    imgplot = plt.imshow(img)
    plt.show()

    testY = testSet["Total tier"]
    testX = testSet.drop("Total tier", axis=1)
    predY = clf.predict(testX)
    
    testX = testX.reset_index()
    prediction = pd.concat([testX['ID'], pd.Series(predY, name="Predicted Tier")], axis=1)
    
    # print("Prediction:\n", prediction.to_string())
    print("Prediction:\n", prediction)
    print("Accuracy of test data: %.2f" % (accuracy_score(testY, predY)))


dataSet = pd.read_csv("D:\Python\Scripts\Exercise\Project\StudentResultsAndResolves.csv")
dataSet = dataSet.rename(columns={"Unnamed: 0": "ID", "race/ethnicity": "Ethnicity"})

newSet = cleanData(dataSet)
print(newSet)

crossTab = pd.crosstab([newSet["Ethnicity"], newSet["ParentEdu_higher education"], newSet["Lunch_standard"], newSet["TestPrep_none"]],
                         newSet["Total tier"])
print(crossTab)

train, test = splitDataSet(newSet)

print("\nTraining Set\n")
print(train)

print("\nTest Set\n")
print(test)

doDecisionTree(train, test)