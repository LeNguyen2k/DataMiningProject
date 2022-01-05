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

#We keeped the data for new purpose, to do the decesion tree to test the change of math, reading and writing



#We clean from math score to writing_guess_max
def cleanData2(data):


    data["parental level of education"] = data["parental level of education"].replace(
        ["bachelor's degree", "college", "master's degree", "associate's degree"], "higher education")

    #print("DataSet:\n", data)

    crossTab = pd.crosstab(
        [data["race/ethnicity"], data["parental level of education"], data["lunch"], data["test preparation course"]],
        data["Total tier"])
    #print("Crosstab:\n", crossTab)

    newSet = data
    #print("newSet:",newSet)
    temp = {"group A": 0, "group B": 1, "group C": 2, "group D": 3, "group E": 4}

    newSet["race/ethnicity"] = newSet["race/ethnicity"].map(temp)
    # newSet = newSet.drop(["Math tier", "Reading tier", "Writing tier"], 1)
    temp = {"higher education":1,"high school":0}
    newSet["parental level of education"] = newSet["parental level of education"].map(temp)
    temp = {"standard":1,"free/reduced":0}
    newSet["lunch"] = newSet["lunch"].map(temp)
    temp = {"completed": 1, "none": 0}
    newSet["test preparation course"] = newSet["test preparation course"].map(temp)
    temp = {"pass":1,"fail":0}
    newSet["Math tier"] = newSet["Math tier"].map(temp)
    newSet["Reading tier"] = newSet["Reading tier"].map(temp)
    newSet["Writing tier"] = newSet["Writing tier"].map(temp)
    newSet["Total tier"] = newSet["Total tier"].map(temp)


    return newSet

def splitDataSet2(NewdataSet):
    train, test = train_test_split(NewdataSet, test_size=0.2, shuffle=False)
    print(f"Train type: {type(train)}")
    return train, test

def doDecisionTree2(trainSet, testSet):

    #Math


    #math time extra study
    trainmathtimeY = trainSet["Extra math study time"]
    trainmathtimeX = trainSet.drop("Extra math study time", axis=1)
    clfmathtime = tree.DecisionTreeClassifier()
    clfmathtime = clfmathtime.fit(trainmathtimeX, trainmathtimeY)
    testmathtimeY = testSet["Extra math study time"]
    testmathtimeX = testSet.drop("Extra math study time", axis=1)
    predmathtimeY = clfmathtime.predict(testmathtimeX)
    testmathX = testmathtimeX.reset_index()
    accuracy_mathtime = accuracy_score(testmathtimeY, predmathtimeY)
    print("Accuracy of predict math test data: %.2f" % (accuracy_mathtime))

    # Reading

    # reading time extra study
    trainreadingtimeY = trainSet["Extra reading study time"]
    trainreadingtimeX = trainSet.drop("Extra reading study time", axis=1)
    clfreadingtime = tree.DecisionTreeClassifier()
    clfreadingtime = clfreadingtime.fit(trainreadingtimeX, trainreadingtimeY)
    testreadingtimeY = testSet["Extra reading study time"]
    testreadingtimeX = testSet.drop("Extra reading study time", axis=1)
    predreadingtimeY = clfreadingtime.predict(testreadingtimeX)
    testreadingX = testreadingtimeX.reset_index()
    accuracy_readingtime = accuracy_score(testreadingtimeY, predreadingtimeY)
    print("Accuracy of predict reading test data: %.2f" % (accuracy_readingtime))

    # writing

    # writing time extra study
    trainwritingtimeY = trainSet["Extra writing study time"]
    trainwritingtimeX = trainSet.drop("Extra writing study time", axis=1)
    clfwritingtime = tree.DecisionTreeClassifier()
    clfwritingtime = clfwritingtime.fit(trainwritingtimeX, trainwritingtimeY)
    testwritingtimeY = testSet["Extra writing study time"]
    testwritingtimeX = testSet.drop("Extra writing study time", axis=1)
    predwritingtimeY = clfwritingtime.predict(testwritingtimeX)
    testwritingX = testwritingtimeX.reset_index()
    accuracy_writingtime = accuracy_score(testwritingtimeY, predwritingtimeY)
    print("Accuracy of predict writing test data: %.2f" % (accuracy_writingtime))

# math guess min


    trainmathminY = trainSet["math_guess_min"]
    trainmathminX = trainSet.drop("math_guess_min", axis=1)
    clfmathmin = tree.DecisionTreeClassifier()
    clfmathmin = clfmathmin.fit(trainmathminX, trainmathminY)
    testmathminY = testSet["math_guess_min"]
    testmathminX = testSet.drop("math_guess_min", axis=1)
    predmathminY = clfmathmin.predict(testmathminX)
    testmathminX = testmathminX.reset_index()
    accuracy_mathmin = accuracy_score(testmathminY, predmathminY)
    print("Accuracy of predict math min in next semester: %.2f" % (accuracy_mathmin))

    trainmathmaxY = trainSet["math_guess_max"]
    trainmathmaxX = trainSet.drop("math_guess_max", axis=1)
    clfmathmax = tree.DecisionTreeClassifier()
    clfmathmax = clfmathmax.fit(trainmathmaxX, trainmathmaxY)
    testmathmaxY = testSet["math_guess_max"]
    testmathmaxX = testSet.drop("math_guess_max", axis=1)
    predmathmaxY = clfmathmax.predict(testmathmaxX)
    testmathmaxX = testmathmaxX.reset_index()
    accuracy_mathmax = accuracy_score(testmathmaxY, predmathmaxY)
    print("Accuracy of predict math max in next semester: %.2f" % (accuracy_mathmax))

# reading guess min


    trainreadingminY = trainSet["reading_guess_min"]
    trainreadingminX = trainSet.drop("reading_guess_min", axis=1)
    clfreadingmin = tree.DecisionTreeClassifier()
    clfreadingmin = clfreadingmin.fit(trainreadingminX, trainreadingminY)
    testreadingminY = testSet["reading_guess_min"]
    testreadingminX = testSet.drop("reading_guess_min", axis=1)
    predreadingminY = clfreadingmin.predict(testreadingminX)
    testreadingminX = testreadingminX.reset_index()
    accuracy_readingmin = accuracy_score(testreadingminY, predreadingminY)
    print("Accuracy of predict reading min in next semester: %.2f" % (accuracy_readingmin))

    trainreadingmaxY = trainSet["reading_guess_max"]
    trainreadingmaxX = trainSet.drop("reading_guess_max", axis=1)
    clfreadingmax = tree.DecisionTreeClassifier()
    clfreadingmax = clfreadingmax.fit(trainreadingmaxX, trainreadingmaxY)
    testreadingmaxY = testSet["reading_guess_max"]
    testreadingmaxX = testSet.drop("reading_guess_max", axis=1)
    predreadingmaxY = clfreadingmax.predict(testreadingmaxX)
    testreadingmaxX = testreadingmaxX.reset_index()
    accuracy_readingmax = accuracy_score(testreadingmaxY, predreadingmaxY)
    print("Accuracy of predict reading max in next semester: %.2f" % (accuracy_readingmax))

    # writing guess min

    trainwritingminY = trainSet["writing_guess_min"]
    trainwritingminX = trainSet.drop("writing_guess_min", axis=1)
    clfwritingmin = tree.DecisionTreeClassifier()
    clfwritingmin = clfwritingmin.fit(trainwritingminX, trainwritingminY)
    testwritingminY = testSet["writing_guess_min"]
    testwritingminX = testSet.drop("writing_guess_min", axis=1)
    predwritingminY = clfwritingmin.predict(testwritingminX)
    testwritingminX = testwritingminX.reset_index()
    accuracy_writingmin = accuracy_score(testwritingminY, predwritingminY)
    print("Accuracy of predict writing min in next semester: %.2f" % (accuracy_writingmin))


    trainwritingmaxY = trainSet["writing_guess_max"]
    trainwritingmaxX = trainSet.drop("writing_guess_max", axis=1)
    clfwritingmax = tree.DecisionTreeClassifier()
    clfwritingmax = clfwritingmax.fit(trainwritingmaxX, trainwritingmaxY)
    testwritingmaxY = testSet["writing_guess_max"]
    testwritingmaxX = testSet.drop("writing_guess_max", axis=1)
    predwritingmaxY = clfwritingmax.predict(testwritingmaxX)
    testwritingmaxX = testwritingmaxX.reset_index()
    accuracy_writingmax = accuracy_score(testwritingmaxY, predwritingmaxY)
    print("Accuracy of predict writing max in next semester: %.2f" % (accuracy_writingmax))


    pred_list = [predmathtimeY.tolist(),predreadingtimeY.tolist(),predwritingtimeY.tolist(),predmathminY.tolist(),predmathmaxY.tolist(),predreadingminY.tolist(),predreadingmaxY.tolist(),predwritingminY.tolist(),predwritingmaxY.tolist()]
    test_list = [testmathtimeY.to_list(),testreadingtimeY.to_list(),testwritingtimeY.to_list(),testmathminY.to_list(),testmathmaxY.to_list(),testreadingminY.to_list(),testreadingmaxY.to_list(),testwritingminY.to_list(),testwritingmaxY.to_list()]
    accuracy_list = [accuracy_mathtime,accuracy_readingtime,accuracy_writingtime,accuracy_mathmin,accuracy_mathmax,accuracy_readingmin,accuracy_readingmax,accuracy_writingmin,accuracy_writingmax]
    return pred_list,test_list,accuracy_list

def Matrix_Generator(matrix):
    for i in range(0,len(matrix)):
        if (i+1)%14 == 0 :
            print(f"{matrix[i]}")
        else:
            if (matrix[i] == 100):
                print(f"{matrix[i]}",end = ' ')
            elif (matrix[i]>=10):
                print(f"{matrix[i]} ",end = ' ')
            else:
                print(f"{matrix[i]}  ",end = " ")
    print("\n")
#Main

NewdataSet = pd.read_csv("StudentResultsAndResolves.csv")
newSet = cleanData2(NewdataSet)




train2,test2 = splitDataSet2(newSet)

#print(train2)
#print(test2)
pred_list,test_list,accuracy_list= doDecisionTree2(train2, test2)
listname = ["math time","reading time","writing time","math min","math max","reading min","reading max","writing min","writing max"]


for i in range(len(pred_list)):
    print(f"Prediction: {listname[i]}: Accuracy: {accuracy_list[i]}\n ")
    Matrix_Generator(pred_list[i])
    print(f"Test: {listname[i]}\n ")
    Matrix_Generator(test_list[i])
