import pandas as pd
import numpy as np
from pandas.core.reshape.pivot import crosstab
from scipy.sparse import data
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.svm import SVC
from sklearn import linear_model

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
    train, test = train_test_split(dataSet, test_size=0.2, shuffle=False)

    return train, test

def doDecisionTree(dataSet):
    tempSet = pd.get_dummies(dataSet, columns=["Total tier"], prefix=["TotalTier"], drop_first=True)
    trainSet, testSet = splitDataSet(tempSet)  
    print(trainSet)

    trainY = trainSet["Class"]
    trainX = trainSet.drop(["Class"], axis=1)
    
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(trainX, trainY)

    dot_data = tree.export_graphviz(clf, feature_names=trainX.columns, class_names=trainSet["Class"], filled=True, out_file=None)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_png("StudentTreeTrain.png")

    img = mpimg.imread('StudentTreeTrain.png')
    imgplot = plt.imshow(img)
    plt.show()

    testY = testSet["Class"]
    testX = testSet.drop("Class", axis=1)
    predY = clf.predict(testX)
    
    testX = testX.reset_index()
    prediction = pd.concat([testX['ID'], pd.Series(predY, name="Predicted Class")], axis=1)
    
    # print("Prediction:\n", prediction.to_string())
    print("Prediction:\n", prediction)
    print("Accuracy of test data: %.2f" % (accuracy_score(testY, predY)))

def KNearestNeighbor(dataset):
    tempSet = pd.get_dummies(dataset, columns=["Total tier"], prefix=["TotalTier"], drop_first=True)
    trainSet, testSet = splitDataSet(tempSet)  
    print(trainSet)

    trainY = trainSet["Class"]
    trainX = trainSet.drop(["Class"], axis=1)

    testY = testSet["Class"]
    testX = testSet.drop("Class", axis=1)

    numNeighbor = [1, 5, 10, 15, 20, 25, 30]

    trainAcc = []
    testAcc = []

    for i in numNeighbor:
        clf = KNeighborsClassifier(n_neighbors= i, metric='minkowski', p=2)
        clf.fit(trainX, trainY)
        predTrainY = clf.predict(trainX)
        predTestY = clf.predict(testX)
        trainAcc.append(accuracy_score(trainY, predTrainY))
        testAcc.append(accuracy_score(testY, predTestY))

    print("Training Accuracy: ", trainAcc)
    print("Testing Accuracy: ", testAcc)

    plt.plot(numNeighbor, trainAcc, 'ro-', numNeighbor, testAcc, 'bv--')
    plt.legend(['Training Accuracy', 'Test Accuracy'])
    plt.xlabel('Number of neighbors')
    plt.ylabel('Accuracy')
    plt.show()

def SVMachine(dataset):
    tempSet = pd.get_dummies(dataset, columns=["Total tier"], prefix=["TotalTier"], drop_first=True)
    trainSet, testSet = splitDataSet(tempSet)  
    print(trainSet)

    trainY = trainSet["Class"]
    trainX = trainSet.drop(["Class"], axis=1)

    testY = testSet["Class"]
    testX = testSet.drop("Class", axis=1)

    C = [0.01, 0.1, 0.2, 0.5, 0.8, 1, 5, 10, 20, 50]
    SVMTrainAcc = []
    SVMTestAcc = []

    for param in C:
        clf = SVC(C=param, kernel='rbf', gamma='auto')
        clf.fit(trainX, trainY)
        predTrainY = clf.predict(trainX)
        predTestY = clf.predict(testX)
        SVMTrainAcc.append(accuracy_score(trainY, predTrainY))
        SVMTestAcc.append(accuracy_score(testY, predTestY))

    plt.plot(C, SVMTrainAcc, 'ro-', C, SVMTestAcc, 'bv--')
    plt.legend(['Training Accuracy', 'Test Accuracy'])
    plt.xlabel('C')
    plt.xscale('log')
    plt.ylabel('Accuracy')
    plt.show()

def logisticRegression(dataset):
    tempSet = pd.get_dummies(dataset, columns=["Total tier"], prefix=["TotalTier"], drop_first=True)
    trainSet, testSet = splitDataSet(tempSet)  
    print(trainSet)

    trainY = trainSet["Class"]
    trainX = trainSet.drop(["Class"], axis=1)

    testY = testSet["Class"]
    testX = testSet.drop("Class", axis=1)

    C = [0.01, 0.1, 0.2, 0.5, 0.8, 1, 5, 10, 20, 50]
    LRtrainAcc = []
    LRtestAcc = []
    SVMTrainAcc = []
    SVMTestAcc = []
    for param in C:
        clf = linear_model.LogisticRegression(C=param)
        clf.fit(trainX, trainY)
        predTrainY = clf.predict(trainX)
        predTestY = clf.predict(testX)
        LRtrainAcc.append(accuracy_score(trainY, predTrainY))
        LRtestAcc.append(accuracy_score(testY, predTestY))

        clf = SVC(C = param, kernel='linear')
        clf.fit(trainX, trainY)
        predTrainY = clf.predict(trainX)
        predTestY = clf.predict(testX)
        SVMTrainAcc.append(accuracy_score(trainY, predTrainY))
        SVMTestAcc.append(accuracy_score(testY, predTestY))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle("Logistic Regression")
    ax1.plot(C, LRtrainAcc, 'ro-', C, LRtestAcc, 'bv--')
    ax1.legend(['Training Accuracy', 'Test Accuracy'])
    ax1.set_xlabel('C')
    ax1.set_xscale('log')
    ax1.set_ylabel('Accuracy')

    ax2.plot(C, SVMTrainAcc, 'ro-', C, SVMTestAcc, 'bv--')
    ax2.legend(['Training Accuracy', 'Test Accuracy'])
    ax2.set_xlabel('C')
    ax2.set_xscale('log')
    ax2.set_ylabel('Accuracy')
    plt.show()

def classLabeling(dataSet):
    dataSet = dataSet.drop(["MathTier_pass", "ReadTier_pass", "WriteTier_pass"], axis=1)

    rowCount = len(dataSet.index)
    print(rowCount)

    temp = dataSet.iloc[0]
    # print(temp)
    # print(temp["Total tier"])

    studentClass = []

    for i in range(rowCount):
        temp = dataSet.iloc[i]

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
            studentClass.append("Unidentified")

    for i in range(len(studentClass)):
        studentClass[i] = studentClass[i].name

    temp = np.array(studentClass)

    dataSet["Class"] = temp

    return dataSet



dataSet = pd.read_csv("D:\Python\Scripts\Exercise\Project\StudentResultsAndResolves.csv")
dataSet = dataSet.rename(columns={"Unnamed: 0": "ID", "race/ethnicity": "Ethnicity"})

newSet = cleanData(dataSet)

crossTab = pd.crosstab([newSet["Ethnicity"], newSet["ParentEdu_higher education"], newSet["Lunch_standard"], newSet["TestPrep_none"]],
                         newSet["Total tier"])
print(crossTab)

newSet = classLabeling(newSet)
print("DATASET:\n")
print(newSet)

# doDecisionTree(newSet)

# KNearestNeighbor(newSet)

# SVMachine(newSet)

logisticRegression(newSet)