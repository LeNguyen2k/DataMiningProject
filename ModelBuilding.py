import pandas as pd
import numpy as np
from pandas.core.reshape.pivot import crosstab
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import pydotplus

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import enum

# 0 -> false
# 1 -> true
#
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
# +) Higher Education -> 1:
#       Associate's degree, 
#       College, 
#       Master's degree, 
#       Bachelor's degree
# +) High school -> 0
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
#
###########################################################################################
# Used enum to set class name easier
# Students are divided into 2 groups: Pass and Fail
# 
# Each has its own sub groups that will be explained below
# 
# Student classification name:
# Passed Exam:
# Ace -> No Test preparation, Free/reduced lunch, Highschool education Parents
# Good -> Completed Test preparation, Free/reduced lunch, Highschool education Parents
# 
# Failed Exam:
# Awful -> No Test preparation, Free/reduced lunch, Highschool education Parents
# Bad -> Completed Test preparation, Free/reduced lunch, Highschool education Parents
# 
# Tails explanation:
# L -> Standard lunch
# P -> Parents have higher education
# PL -> Standard lunch, Parents have higher education

class PassGroup(enum.Enum):
    Ace = 1
    Good = 2
    Ace_L = 3
    Good_L = 4
    Ace_P = 5
    Good_P = 6
    Ace_PL = 7
    Good_PL = 8

class FailGroup(enum.Enum):
    Awful = 1
    Bad = 2
    Awful_L = 3
    Bad_L = 4
    Awful_P = 5
    Bad_P = 6
    Awful_PL = 7
    Bad_PL = 8

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

def ruleBased(trainSet, testSet):
    train = trainSet.drop(["MathTier_pass", "ReadTier_pass", "WriteTier_pass"], axis=1)
    test = testSet.drop(["MathTier_pass", "ReadTier_pass", "WriteTier_pass"], axis=1)

    rowCount = len(train.index)
    print(rowCount)

    temp = train.iloc[0]
    # print(temp)
    # print(temp["Total tier"])

    studentClass = []

    for i in range(rowCount):
        temp = train.iloc[i]

        # Classifying students that passed the exam
        if temp["Total tier"] == "pass" and temp["ParentEdu_higher education"] == 0 and temp["Lunch_standard"] == 0 and temp["TestPrep_none"] == 0:
            studentClass.append(PassGroup.Good)
        elif temp["Total tier"] == "pass" and temp["ParentEdu_higher education"] == 0 and temp["Lunch_standard"] == 0 and temp["TestPrep_none"] == 1:
            studentClass.append(PassGroup.Ace)
        elif temp["Total tier"] == "pass" and temp["ParentEdu_higher education"] == 0 and temp["Lunch_standard"] == 1 and temp["TestPrep_none"] == 0:
            studentClass.append(PassGroup.Good_L)
        elif temp["Total tier"] == "pass" and temp["ParentEdu_higher education"] == 0 and temp["Lunch_standard"] == 1 and temp["TestPrep_none"] == 1:
            studentClass.append(PassGroup.Ace_L)
        elif temp["Total tier"] == "pass" and temp["ParentEdu_higher education"] == 1 and temp["Lunch_standard"] == 0 and temp["TestPrep_none"] == 0:
            studentClass.append(PassGroup.Good_P)
        elif temp["Total tier"] == "pass" and temp["ParentEdu_higher education"] == 1 and temp["Lunch_standard"] == 0 and temp["TestPrep_none"] == 1:
            studentClass.append(PassGroup.Ace_P)
        elif temp["Total tier"] == "pass" and temp["ParentEdu_higher education"] == 1 and temp["Lunch_standard"] == 1 and temp["TestPrep_none"] == 0:
            studentClass.append(PassGroup.Good_PL)
        elif temp["Total tier"] == "pass" and temp["ParentEdu_higher education"] == 1 and temp["Lunch_standard"] == 1 and temp["TestPrep_none"] == 1:
            studentClass.append(PassGroup.Ace_PL)

        # Classifying students that failed the exam
        elif temp["Total tier"] == "fail" and temp["ParentEdu_higher education"] == 0 and temp["Lunch_standard"] == 0 and temp["TestPrep_none"] == 0:
            studentClass.append(FailGroup.Bad)
        elif temp["Total tier"] == "fail" and temp["ParentEdu_higher education"] == 0 and temp["Lunch_standard"] == 0 and temp["TestPrep_none"] == 1:
            studentClass.append(FailGroup.Awful)
        elif temp["Total tier"] == "fail" and temp["ParentEdu_higher education"] == 0 and temp["Lunch_standard"] == 1 and temp["TestPrep_none"] == 0:
            studentClass.append(FailGroup.Bad_L)
        elif temp["Total tier"] == "fail" and temp["ParentEdu_higher education"] == 0 and temp["Lunch_standard"] == 1 and temp["TestPrep_none"] == 1:
            studentClass.append(FailGroup.Awful_L)
        elif temp["Total tier"] == "fail" and temp["ParentEdu_higher education"] == 1 and temp["Lunch_standard"] == 0 and temp["TestPrep_none"] == 0:
            studentClass.append(FailGroup.Bad_P)
        elif temp["Total tier"] == "fail" and temp["ParentEdu_higher education"] == 1 and temp["Lunch_standard"] == 0 and temp["TestPrep_none"] == 1:
            studentClass.append(FailGroup.Awful_P)
        elif temp["Total tier"] == "fail" and temp["ParentEdu_higher education"] == 1 and temp["Lunch_standard"] == 1 and temp["TestPrep_none"] == 0:
            studentClass.append(FailGroup.Bad_PL)
        elif temp["Total tier"] == "fail" and temp["ParentEdu_higher education"] == 1 and temp["Lunch_standard"] == 1 and temp["TestPrep_none"] == 1:
            studentClass.append(FailGroup.Awful_PL)
        else:
            studentClass.append("Error")

    for i in range(len(studentClass)):
        studentClass[i] = studentClass[i].name

    temp = np.array(studentClass)

    aceCount = (temp=="Ace").sum()
    print("Number of Ace: ", aceCount)
    goodCount = (temp=="Good").sum()
    print("Number of Good: ", goodCount)

    awfulCount = (temp=="Awful").sum()
    print("Number of Awful: ", awfulCount)
    badCount = (temp=="Bad").sum()
    print("Number of Bad: ", badCount)

    coverage1 = aceCount / len(train.index)
    print("Coverage of rule 1: ", coverage1)
    coverage2 = goodCount / len(train.index)
    print("Coverage of rule 2: ", coverage2)
    
    # train["Class"] = studentClass
    # print(train)

    # groupedObj = train.groupby(["Total tier", "Ethnicity"])
    # for key, item in groupedObj:
    #     print("Key: " + str(key))
    #     print(str(item), "\n\n")


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

# doDecisionTree(train, test)

ruleBased(train, test)

# print(type(PassGroup.Ace.name))
# print(PassGroup.Ace.name)
