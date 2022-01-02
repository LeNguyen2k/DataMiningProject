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

#We clean from math score to writing_guess_max
def cleanData(dataSet):
    attribute_To_Drop = ["math score", "reading score", "writing score", "Extra math study time",
                         "Extra reading study time", "Extra writing study time",
                         "math_guess_min", "math_guess_max", "reading_guess_min", "reading_guess_max",
                         "writing_guess_min", "writing_guess_max"]

    data = dataSet.drop(attribute_To_Drop,axis =  1)

    data["parental level of education"] = data["parental level of education"].replace(
        ["bachelor's degree", "college", "master's degree", "associate's degree"], "higher education")

    print("DataSet:\n", data)

    crossTab = pd.crosstab(
        [data["Ethnicity"], data["parental level of education"], data["lunch"], data["test preparation course"]],
        data["Total tier"])
    print(crossTab)

    newSet = pd.get_dummies(data, columns=["Math tier", "Reading tier", "Writing tier", "parental level of education",
                                           "lunch", "test preparation course"],
                            prefix=["MathTier", "ReadTier", "WriteTier", "ParentEdu", "Lunch", "TestPrep"],
                            drop_first=True)

    temp = {"group A": 0, "group B": 1, "group C": 2, "group D": 3, "group E": 4}

    newSet["Ethnicity"] = newSet["Ethnicity"].map(temp)
    # newSet = newSet.drop(["Math tier", "Reading tier", "Writing tier"], 1)

    print("CLEANED DATASET:\n", newSet)

    return newSet

# We take 80% train, 20 % test
def splitDataSet(dataSet):
    train, test = train_test_split(newSet, test_size=0.2, shuffle=False)

    return train, test

# Do DecisionTree for training the data
def doDecisionTree(trainSet, testSet):
    trainY = trainSet["Total tier"]
    trainX = trainSet.drop("Total tier", axis=1)

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(trainX, trainY)

    dot_data = tree.export_graphviz(clf, feature_names=trainX.columns, class_names=["Fail", "Pass"], filled=True,
                                    out_file=None)
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
    print("Prediction:\n", prediction["Predicted Tier"])
    print("Accuracy of test data: %.2f" % (accuracy_score(testY, predY)))
    TrainPass = (trainY == "pass").sum()
    TrainFail = (trainY == "fail").sum()
    TestPass = (prediction["Predicted Tier"] == "pass").sum()
    TestFail = (prediction["Predicted Tier"] == "fail").sum()
    print(f"Pass and Fail in Train and Test:\n Train: {TrainPass} , {TrainFail}\n Test: {TestPass} , {TestFail}")
    plotbarscore = plt.figure("Training data for Student table")

    TrainTotaltierBar = ["Pass","Fail"]
    TrainTotaltierQuantity = [TrainPass,TrainFail]
    plt.subplot(2, 1, 1)
    plt.bar(TrainTotaltierBar, TrainTotaltierQuantity)
    plt.title('Training set  for Total tier')
    plt.xlabel('Training set')
    plt.ylabel('Number of Student')

    TestTotaltierBar = ["Pass", "Fail"]
    TestTotaltierQuantity = [TestPass, TestFail]
    plt.subplot(2, 1, 2)
    plt.bar(TestTotaltierBar, TestTotaltierQuantity)
    plt.title('Testing set  for Total tier')
    plt.xlabel('Testing set')
    plt.ylabel('Number of Student')
    plt.show()

def ruleBased(trainSet, testSet):
    train = trainSet.drop(["MathTier_pass", "ReadTier_pass", "WriteTier_pass"], axis=1)
    test = testSet.drop(["MathTier_pass", "ReadTier_pass", "WriteTier_pass"], axis=1)


    rowCounttrain = len(train.index)
    rowCounttest = len(test.index)
    print(rowCounttrain)

    temp = train.iloc[0]
    temptest = test.iloc[0]
    # print(temp)
    # print(temp["Total tier"])

    studentClass = []
    studentClasstest = []
    for i in range(rowCounttrain):
        temp = train.iloc[i]
        PassStudent = (temp["Total tier"] == "pass")
        FailStudent = (temp["Total tier"] == "fail")
        # Classifying students that passed the exam
        if PassStudent and temp["ParentEdu_higher education"] == 0 and temp["Lunch_standard"] == 0 and \
                temp["TestPrep_none"] == 0:
            studentClass.append(PassGroup.Good)
        elif PassStudent and temp["ParentEdu_higher education"] == 0 and temp[
            "Lunch_standard"] == 0 and temp["TestPrep_none"] == 1:
            studentClass.append(PassGroup.Ace)
        elif PassStudent and temp["ParentEdu_higher education"] == 0 and temp[
            "Lunch_standard"] == 1 and temp["TestPrep_none"] == 0:
            studentClass.append(PassGroup.Good_L)
        elif PassStudent and temp["ParentEdu_higher education"] == 0 and temp[
            "Lunch_standard"] == 1 and temp["TestPrep_none"] == 1:
            studentClass.append(PassGroup.Ace_L)
        elif PassStudent and temp["ParentEdu_higher education"] == 1 and temp[
            "Lunch_standard"] == 0 and temp["TestPrep_none"] == 0:
            studentClass.append(PassGroup.Good_P)
        elif PassStudent and temp["ParentEdu_higher education"] == 1 and temp[
            "Lunch_standard"] == 0 and temp["TestPrep_none"] == 1:
            studentClass.append(PassGroup.Ace_P)
        elif PassStudent and temp["ParentEdu_higher education"] == 1 and temp[
            "Lunch_standard"] == 1 and temp["TestPrep_none"] == 0:
            studentClass.append(PassGroup.Good_PL)
        elif PassStudent and temp["ParentEdu_higher education"] == 1 and temp[
            "Lunch_standard"] == 1 and temp["TestPrep_none"] == 1:
            studentClass.append(PassGroup.Ace_PL)


        # Classifying students that failed the exam
        elif FailStudent and temp["ParentEdu_higher education"] == 0 and temp[
            "Lunch_standard"] == 0 and temp["TestPrep_none"] == 0:
            studentClass.append(FailGroup.Bad)
        elif FailStudent and temp["ParentEdu_higher education"] == 0 and temp[
            "Lunch_standard"] == 0 and temp["TestPrep_none"] == 1:
            studentClass.append(FailGroup.Awful)
        elif FailStudent and temp["ParentEdu_higher education"] == 0 and temp[
            "Lunch_standard"] == 1 and temp["TestPrep_none"] == 0:
            studentClass.append(FailGroup.Bad_L)
        elif FailStudent and temp["ParentEdu_higher education"] == 0 and temp[
            "Lunch_standard"] == 1 and temp["TestPrep_none"] == 1:
            studentClass.append(FailGroup.Awful_L)
        elif FailStudent and temp["ParentEdu_higher education"] == 1 and temp[
            "Lunch_standard"] == 0 and temp["TestPrep_none"] == 0:
            studentClass.append(FailGroup.Bad_P)
        elif FailStudent and temp["ParentEdu_higher education"] == 1 and temp[
            "Lunch_standard"] == 0 and temp["TestPrep_none"] == 1:
            studentClass.append(FailGroup.Awful_P)
        elif FailStudent and temp["ParentEdu_higher education"] == 1 and temp[
            "Lunch_standard"] == 1 and temp["TestPrep_none"] == 0:
            studentClass.append(FailGroup.Bad_PL)
        elif FailStudent and temp["ParentEdu_higher education"] == 1 and temp[
            "Lunch_standard"] == 1 and temp["TestPrep_none"] == 1:
            studentClass.append(FailGroup.Awful_PL)
        else:
            studentClass.append("Error")

    for i in range(len(studentClass)):
        studentClass[i] = studentClass[i].name



    temp = np.array(studentClass)
    print(f"Train index = {len(train.index)}")
    aceCount = (temp == "Ace").sum()
    print(f"Number of Ace: {aceCount}")
    goodCount = (temp == "Good").sum()
    print(f"Number of Good: {goodCount}")
    ace_with_lunchCount = (temp == "Ace_L").sum()
    print(f"Number of Ace with lunch: {ace_with_lunchCount}")
    good_with_lunchCount = (temp == "Good_L").sum()
    print(f"Number of Good with lunch: {good_with_lunchCount}")
    ace_with_highparentCount = (temp == "Ace_P").sum()
    print(f"Number of Ace with high parent: {ace_with_highparentCount}")
    good_with_highparentCount = (temp == "Good_P").sum()
    print(f"Number of Good with high parent: {good_with_highparentCount}")
    ace_with_lunch_and_highparentCount = (temp == "Ace_PL").sum()
    print(f"Number of Ace with lunch and high parent: {ace_with_lunch_and_highparentCount}")
    good_with_lunch_and_highparentCount = (temp == "Good_PL").sum()
    print(f"Number of Good with lunch and high parent: {good_with_lunch_and_highparentCount}\n\n")

    awfulCount = (temp == "Awful").sum()
    print(f"Number of Awful: {awfulCount}")
    badCount = (temp == "Bad").sum()
    print(f"Number of Bad: {badCount}")
    awful_with_lunchCount = (temp == "Awful_L").sum()
    print(f"Number of Awful with lunch: {awful_with_lunchCount}")
    bad_with_lunchCount = (temp == "Bad_L").sum()
    print(f"Number of Bad with lunch: {bad_with_lunchCount}")
    awful_with_highparentCount = (temp == "Awful_P").sum()
    print(f"Number of awful with high parent: {awful_with_highparentCount}")
    bad_with_highparentCount = (temp == "Bad_P").sum()
    print(f"Number of bad with high parent: {bad_with_highparentCount}")
    awful_with_lunch_and_highparentCount = (temp == "Awful_PL").sum()
    print(f"Number of awful with lunch and high parent: {awful_with_lunch_and_highparentCount}")
    bad_with_lunch_and_highparentCount = (temp == "Bad_PL").sum()
    print(f"Number of bad with lunch and high parent: {bad_with_lunch_and_highparentCount}\n\n")

    acecoverage = aceCount * 100 / len(train.index)
    print(f"Coverage of ace Student: {acecoverage}")
    goodcoverage = goodCount * 100 / len(train.index)
    print(f"Coverage of good Student: {goodcoverage}")
    ace_with_lunchcoverage = ace_with_lunchCount * 100 / len(train.index)
    print(f"Coverage of ace with lunch Student: {ace_with_lunchcoverage}")
    good_with_lunchcoverage = good_with_lunchCount * 100 / len(train.index)
    print(f"Coverage of good with lunch Student: {good_with_lunchcoverage}")
    ace_with_highparentcoverage = ace_with_highparentCount * 100 / len(train.index)
    print(f"Coverage of ace with high parent Student: {ace_with_highparentcoverage}")
    good_with_highparentcoverage = good_with_highparentCount * 100 / len(train.index)
    print(f"Coverage of good with high parent Student: {good_with_highparentcoverage}")
    ace_with_bothcoverage = ace_with_lunch_and_highparentCount * 100 / len(train.index)
    print(f"Coverage of ace with lunch and high parent Student: {ace_with_bothcoverage}")
    good_with_bothcoverage = good_with_lunch_and_highparentCount * 100 / len(train.index)
    print(f"Coverage of good with lunch and high parent Student: {good_with_bothcoverage}\n\n")

    awfulcoverage = awfulCount * 100 / len(train.index)
    print(f"Coverage of awful Student: {awfulcoverage}")
    badcoverage = badCount * 100 / len(train.index)
    print(f"Coverage of bad Student: {badcoverage}")
    awful_with_lunchcoverage = awful_with_lunchCount * 100 / len(train.index)
    print(f"Coverage of awful with lunch Student: {awful_with_lunchcoverage}")
    bad_with_lunchcoverage = bad_with_lunchCount * 100 / len(train.index)
    print(f"Coverage of bad with lunch Student: {bad_with_lunchcoverage}")
    awful_with_highparentcoverage = awful_with_highparentCount * 100 / len(train.index)
    print(f"Coverage of awful with high parent Student: {awful_with_highparentcoverage}")
    bad_with_highparentcoverage = bad_with_highparentCount * 100 / len(train.index)
    print(f"Coverage of bad with high parent Student: {bad_with_highparentcoverage}")
    awful_with_bothcoverage = awful_with_lunch_and_highparentCount * 100 / len(train.index)
    print(f"Coverage of awful with lunch and high parent Student: {awful_with_bothcoverage}")
    bad_with_bothcoverage = bad_with_lunch_and_highparentCount * 100 / len(train.index)
    print(f"Coverage of bad with lunch and high parent Student: {bad_with_bothcoverage}\n\n")

    plotbarscore = plt.figure("Training data for Student result based on some conditions")

    PassStudentbar = ["Ace", "Good", "Ace with Lunch", "Good with lunch", "Ace with high parent",
                      "Good with high parent", "Ace with both", "Good with both"]
    PassStudentquantity = [aceCount, goodCount, ace_with_lunchCount, good_with_lunchCount, ace_with_highparentCount,
                           good_with_highparentCount, ace_with_lunch_and_highparentCount,
                           good_with_lunch_and_highparentCount]
    plt.subplot(2, 1, 1)
    plt.bar(PassStudentbar, PassStudentquantity)
    plt.title('Pass student show')
    plt.xlabel('Pass typed')
    plt.ylabel('Number of Student')

    FailStudentbar = ["Awful", "Bad", "Awful with Lunch", "Bad with lunch", "Awful with high parent",
                      "Bad with high parent", "Awful with both", "Bad with both"]
    FailStudentquantity = [awfulCount, badCount, awful_with_lunchCount, bad_with_lunchCount, awful_with_highparentCount,
                           bad_with_highparentCount, awful_with_lunch_and_highparentCount,
                           bad_with_lunch_and_highparentCount]
    plt.subplot(2, 1, 2)
    plt.bar(FailStudentbar, FailStudentquantity)
    plt.title('Fail student show')
    plt.xlabel('Fail typed')
    plt.ylabel('Number of Student')
    plt.show()

    plotbarscore = plt.figure("Coverage data training displayed by percentage")

    PassStudentcoveragebar = ["Ace", "Good", "Ace with Lunch", "Good with lunch", "Ace with high parent",
                              "Good with high parent", "Ace with both", "Good with both"]
    PassStudentquantity = [acecoverage, goodcoverage, ace_with_lunchcoverage, good_with_lunchcoverage,
                           ace_with_highparentcoverage, good_with_highparentcoverage, ace_with_bothcoverage,
                           good_with_bothcoverage]
    plt.subplot(2, 1, 1)
    plt.bar(PassStudentbar, PassStudentquantity)
    plt.title('Pass student show')
    plt.xlabel('Pass typed')
    plt.ylabel('Percentage')

    # Draw bar chart for Reading

    FailStudentbar = ["Awful", "Bad", "Awful with Lunch", "Bad with lunch", "Awful with high parent",
                      "Bad with high parent", "Awful with both", "Bad with both"]
    FailStudentquantity = [awfulcoverage, badcoverage, awful_with_lunchcoverage, bad_with_lunchcoverage,
                           awful_with_highparentcoverage, bad_with_highparentcoverage, awful_with_bothcoverage,
                           bad_with_bothcoverage]
    plt.subplot(2, 1, 2)
    plt.bar(FailStudentbar, FailStudentquantity)
    plt.title('Fail student show')
    plt.xlabel('Fail typed')
    plt.ylabel('Percentage')
    plt.show()

    for i in range(rowCounttest):
        temptest = test.iloc[i]
        PassStudent = (temptest["Total tier"] == "pass")
        FailStudent = (temptest["Total tier"] == "fail")
        # Classifying students that passed the exam
        if PassStudent and temptest["ParentEdu_higher education"] == 0 and temptest["Lunch_standard"] == 0 and \
                temptest["TestPrep_none"] == 0:
            studentClasstest.append(PassGroup.Good)
        elif PassStudent and temptest["ParentEdu_higher education"] == 0 and temptest[
            "Lunch_standard"] == 0 and temptest["TestPrep_none"] == 1:
            studentClasstest.append(PassGroup.Ace)
        elif PassStudent and temptest["ParentEdu_higher education"] == 0 and temptest[
            "Lunch_standard"] == 1 and temptest["TestPrep_none"] == 0:
            studentClasstest.append(PassGroup.Good_L)
        elif PassStudent and temptest["ParentEdu_higher education"] == 0 and temptest[
            "Lunch_standard"] == 1 and temptest["TestPrep_none"] == 1:
            studentClasstest.append(PassGroup.Ace_L)
        elif PassStudent and temptest["ParentEdu_higher education"] == 1 and temptest[
            "Lunch_standard"] == 0 and temptest["TestPrep_none"] == 0:
            studentClasstest.append(PassGroup.Good_P)
        elif PassStudent and temptest["ParentEdu_higher education"] == 1 and temptest[
            "Lunch_standard"] == 0 and temptest["TestPrep_none"] == 1:
            studentClasstest.append(PassGroup.Ace_P)
        elif PassStudent and temptest["ParentEdu_higher education"] == 1 and temptest[
            "Lunch_standard"] == 1 and temptest["TestPrep_none"] == 0:
            studentClasstest.append(PassGroup.Good_PL)
        elif PassStudent and temptest["ParentEdu_higher education"] == 1 and temptest[
            "Lunch_standard"] == 1 and temptest["TestPrep_none"] == 1:
            studentClasstest.append(PassGroup.Ace_PL)


        # Classifying students that failed the exam
        elif FailStudent and temptest["ParentEdu_higher education"] == 0 and temptest[
            "Lunch_standard"] == 0 and temptest["TestPrep_none"] == 0:
            studentClasstest.append(FailGroup.Bad)
        elif FailStudent and temptest["ParentEdu_higher education"] == 0 and temptest[
            "Lunch_standard"] == 0 and temptest["TestPrep_none"] == 1:
            studentClasstest.append(FailGroup.Awful)
        elif FailStudent and temptest["ParentEdu_higher education"] == 0 and temptest[
            "Lunch_standard"] == 1 and temptest["TestPrep_none"] == 0:
            studentClasstest.append(FailGroup.Bad_L)
        elif FailStudent and temptest["ParentEdu_higher education"] == 0 and temptest[
            "Lunch_standard"] == 1 and temptest["TestPrep_none"] == 1:
            studentClasstest.append(FailGroup.Awful_L)
        elif FailStudent and temptest["ParentEdu_higher education"] == 1 and temptest[
            "Lunch_standard"] == 0 and temptest["TestPrep_none"] == 0:
            studentClasstest.append(FailGroup.Bad_P)
        elif FailStudent and temptest["ParentEdu_higher education"] == 1 and temptest[
            "Lunch_standard"] == 0 and temptest["TestPrep_none"] == 1:
            studentClasstest.append(FailGroup.Awful_P)
        elif FailStudent and temptest["ParentEdu_higher education"] == 1 and temptest[
            "Lunch_standard"] == 1 and temptest["TestPrep_none"] == 0:
            studentClasstest.append(FailGroup.Bad_PL)
        elif FailStudent and temptest["ParentEdu_higher education"] == 1 and temptest[
            "Lunch_standard"] == 1 and temptest["TestPrep_none"] == 1:
            studentClasstest.append(FailGroup.Awful_PL)
        else:
            studentClasstest.append("Error")
    for i in range(len(studentClasstest)):
        studentClasstest[i] = studentClasstest[i].name

    temptest = np.array(studentClasstest)
    print(f"test index = {len(test.index)}")
    aceCount = (temptest == "Ace").sum()
    print(f"Number of Ace: {aceCount}")
    goodCount = (temptest == "Good").sum()
    print(f"Number of Good: {goodCount}")
    ace_with_lunchCount = (temptest == "Ace_L").sum()
    print(f"Number of Ace with lunch: {ace_with_lunchCount}")
    good_with_lunchCount = (temptest == "Good_L").sum()
    print(f"Number of Good with lunch: {good_with_lunchCount}")
    ace_with_highparentCount = (temptest == "Ace_P").sum()
    print(f"Number of Ace with high parent: {ace_with_highparentCount}")
    good_with_highparentCount = (temptest == "Good_P").sum()
    print(f"Number of Good with high parent: {good_with_highparentCount}")
    ace_with_lunch_and_highparentCount = (temptest == "Ace_PL").sum()
    print(f"Number of Ace with lunch and high parent: {ace_with_lunch_and_highparentCount}")
    good_with_lunch_and_highparentCount = (temptest == "Good_PL").sum()
    print(f"Number of Good with lunch and high parent: {good_with_lunch_and_highparentCount}\n\n")

    awfulCount = (temptest == "Awful").sum()
    print(f"Number of Awful: {awfulCount}")
    badCount = (temptest == "Bad").sum()
    print(f"Number of Bad: {badCount}")
    awful_with_lunchCount = (temptest == "Awful_L").sum()
    print(f"Number of Awful with lunch: {awful_with_lunchCount}")
    bad_with_lunchCount = (temptest == "Bad_L").sum()
    print(f"Number of Bad with lunch: {bad_with_lunchCount}")
    awful_with_highparentCount = (temptest == "Awful_P").sum()
    print(f"Number of awful with high parent: {awful_with_highparentCount}")
    bad_with_highparentCount = (temptest == "Bad_P").sum()
    print(f"Number of bad with high parent: {bad_with_highparentCount}")
    awful_with_lunch_and_highparentCount = (temptest == "Awful_PL").sum()
    print(f"Number of awful with lunch and high parent: {awful_with_lunch_and_highparentCount}")
    bad_with_lunch_and_highparentCount = (temptest == "Bad_PL").sum()
    print(f"Number of bad with lunch and high parent: {bad_with_lunch_and_highparentCount}\n\n")

    acecoverage = aceCount * 100 / len(test.index)
    print(f"Coverage of ace Student: {acecoverage}")
    goodcoverage = goodCount * 100 / len(test.index)
    print(f"Coverage of good Student: {goodcoverage}")
    ace_with_lunchcoverage = ace_with_lunchCount * 100 / len(test.index)
    print(f"Coverage of ace with lunch Student: {ace_with_lunchcoverage}")
    good_with_lunchcoverage = good_with_lunchCount * 100 / len(test.index)
    print(f"Coverage of good with lunch Student: {good_with_lunchcoverage}")
    ace_with_highparentcoverage = ace_with_highparentCount * 100 / len(test.index)
    print(f"Coverage of ace with high parent Student: {ace_with_highparentcoverage}")
    good_with_highparentcoverage = good_with_highparentCount * 100 / len(test.index)
    print(f"Coverage of good with high parent Student: {good_with_highparentcoverage}")
    ace_with_bothcoverage = ace_with_lunch_and_highparentCount * 100 / len(test.index)
    print(f"Coverage of ace with lunch and high parent Student: {ace_with_bothcoverage}")
    good_with_bothcoverage = good_with_lunch_and_highparentCount * 100 / len(test.index)
    print(f"Coverage of good with lunch and high parent Student: {good_with_bothcoverage}\n\n")

    awfulcoverage = awfulCount * 100 / len(test.index)
    print(f"Coverage of awful Student: {awfulcoverage}")
    badcoverage = badCount * 100 / len(test.index)
    print(f"Coverage of bad Student: {badcoverage}")
    awful_with_lunchcoverage = awful_with_lunchCount * 100 / len(test.index)
    print(f"Coverage of awful with lunch Student: {awful_with_lunchcoverage}")
    bad_with_lunchcoverage = bad_with_lunchCount * 100 / len(test.index)
    print(f"Coverage of bad with lunch Student: {bad_with_lunchcoverage}")
    awful_with_highparentcoverage = awful_with_highparentCount * 100 / len(test.index)
    print(f"Coverage of awful with high parent Student: {awful_with_highparentcoverage}")
    bad_with_highparentcoverage = bad_with_highparentCount * 100 / len(test.index)
    print(f"Coverage of bad with high parent Student: {bad_with_highparentcoverage}")
    awful_with_bothcoverage = awful_with_lunch_and_highparentCount * 100 / len(test.index)
    print(f"Coverage of awful with lunch and high parent Student: {awful_with_bothcoverage}")
    bad_with_bothcoverage = bad_with_lunch_and_highparentCount * 100 / len(test.index)
    print(f"Coverage of bad with lunch and high parent Student: {bad_with_bothcoverage}\n\n")

    plotbarscore = plt.figure("Testing data for Student result based on some conditions")

    PassStudentbar = ["Ace", "Good", "Ace with Lunch", "Good with lunch", "Ace with high parent",
                      "Good with high parent", "Ace with both", "Good with both"]
    PassStudentquantity = [aceCount, goodCount, ace_with_lunchCount, good_with_lunchCount, ace_with_highparentCount,
                           good_with_highparentCount, ace_with_lunch_and_highparentCount,
                           good_with_lunch_and_highparentCount]
    plt.subplot(2, 1, 1)
    plt.bar(PassStudentbar, PassStudentquantity)
    plt.title('Pass student show')
    plt.xlabel('Pass typed')
    plt.ylabel('Number of Student')

    FailStudentbar = ["Awful", "Bad", "Awful with Lunch", "Bad with lunch", "Awful with high parent",
                      "Bad with high parent", "Awful with both", "Bad with both"]
    FailStudentquantity = [awfulCount, badCount, awful_with_lunchCount, bad_with_lunchCount, awful_with_highparentCount,
                           bad_with_highparentCount, awful_with_lunch_and_highparentCount,
                           bad_with_lunch_and_highparentCount]
    plt.subplot(2, 1, 2)
    plt.bar(FailStudentbar, FailStudentquantity)
    plt.title('Fail student show')
    plt.xlabel('Fail typed')
    plt.ylabel('Number of Student')
    plt.show()

    plotbarscore = plt.figure("Coverage data testing displayed by percentage")

    PassStudentcoveragebar = ["Ace", "Good", "Ace with Lunch", "Good with lunch", "Ace with high parent",
                              "Good with high parent", "Ace with both", "Good with both"]
    PassStudentquantity = [acecoverage, goodcoverage, ace_with_lunchcoverage, good_with_lunchcoverage,
                           ace_with_highparentcoverage, good_with_highparentcoverage, ace_with_bothcoverage,
                           good_with_bothcoverage]
    plt.subplot(2, 1, 1)
    plt.bar(PassStudentbar, PassStudentquantity)
    plt.title('Pass student show')
    plt.xlabel('Pass typed')
    plt.ylabel('Percentage')

    # Draw bar chart for Reading

    FailStudentbar = ["Awful", "Bad", "Awful with Lunch", "Bad with lunch", "Awful with high parent",
                      "Bad with high parent", "Awful with both", "Bad with both"]
    FailStudentquantity = [awfulcoverage, badcoverage, awful_with_lunchcoverage, bad_with_lunchcoverage,
                           awful_with_highparentcoverage, bad_with_highparentcoverage, awful_with_bothcoverage,
                           bad_with_bothcoverage]
    plt.subplot(2, 1, 2)
    plt.bar(FailStudentbar, FailStudentquantity)
    plt.title('Fail student show')
    plt.xlabel('Fail typed')
    plt.ylabel('Percentage')
    plt.show()
    # train["Class"] = studentClass
    # print(train)

    # groupedObj = train.groupby(["Total tier", "Ethnicity"])
    # for key, item in groupedObj:
    #     print("Key: " + str(key))
    #     print(str(item), "\n\n")


dataSet = pd.read_csv("StudentResultsAndResolves.csv")
dataSet = dataSet.rename(columns={"Unnamed: 0": "ID", "race/ethnicity": "Ethnicity"})

newSet = cleanData(dataSet)
print(f"The new set is \n {newSet}")

crossTab = pd.crosstab(
    [newSet["Ethnicity"], newSet["ParentEdu_higher education"], newSet["Lunch_standard"], newSet["TestPrep_none"]],
    newSet["Total tier"])
print(f"The new crossTab is \n{crossTab}")

train, test = splitDataSet(newSet)

print("\nTraining Set\n")
print(train)

print("\nTest Set\n")
print(test)

doDecisionTree(train, test)

ruleBased(train, test)


# print(type(PassGroup.Ace.name))
# print(PassGroup.Ace.name)

