import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

# Starting time / Evaluate the running time

start_time = time.time()


# Input data


def average(n):
    return sum(n) / (len(n))


def deviation(n):
    aver = average(n)
    s = sum((i - aver) ** 2 for i in n)
    s = (s / (len(n) - 1)) ** 0.5
    return s



input_data = pd.read_csv('StudentsPerformance.csv')
data = input_data
for x in data.index:
    if data.loc[x,"parental level of education"] == "some high school":
        data.loc[x,"parental level of education"] = "high school"
    if data.loc[x,"parental level of education"] == "some college":
        data.loc[x,"parental level of education"] = "college"
data.drop('gender', inplace=True, axis=1)
print(data['parental level of education'])
print(data)
# Math pass (yes if >=50, no if <50)

conditions1 = [(data['math score'] >= 50), (data['math score'] < 50)]
values1 = ['pass', 'fail']
data['Math tier'] = np.select(conditions1, values1)

# Reading pass (yes if >=50, no if <50)

conditions2 = [(data['reading score'] >= 50), (data['reading score'] < 50)]
values2 = ['pass', 'fail']
data['Reading tier'] = np.select(conditions2, values2)

# Writing pass (yes if >=50, no if <50)

conditions3 = [(data['writing score'] >= 50), (data['writing score'] < 50)]
values3 = ['pass', 'fail']
data['Writing tier'] = np.select(conditions3, values3)

# Total pass

conditions = [((data['math score'] >= 50) & (data['reading score'] >= 50) & (data['writing score'] >= 50)),
              ((data['math score'] < 50) | (data['reading score'] < 50) | (data['writing score'] < 50))]
values = ['pass', 'fail']
data['Total tier'] = np.select(conditions, values)
# <50 100% extra class
# 50-69:(5-95% go to extra class)
# >=70: Doesn't have to go to extra class
# Probability that have to study in math
conditions_math = [(data['math score'] < 50),
                   (data['math score'] >= 70),
                   (data['math score'] >= 50) & (data['math score'] < 70)]
values_math = [100, 0, round(100 - (data['math score'] - 49) * (100 / 22))]
data['Extra study math percentage'] = np.select(conditions_math, values_math)

# Probability that have to study in reading
conditions_reading = [(data['reading score'] < 50),
                      (data['reading score'] >= 70),
                      (data['reading score'] >= 50) & (data['reading score'] < 70)]
values_reading = [100, 0, round(100 - (data['reading score'] - 49) * 100 / 22)]
data['Extra study reading percentage'] = np.select(conditions_reading, values_reading)

# Probability that have to study in writing
conditions_writing = [(data['writing score'] < 50),
                      (data['writing score'] >= 70),
                      (data['writing score'] >= 50) & (data['writing score'] < 70)]
values_writing = [100, 0, round(100 - (data['writing score'] - 49) * 100 / 22)]
data['Extra study writing percentage'] = np.select(conditions_writing, values_writing)

# Guessing point from next exam with range from min to max base on present point


# Math
math_guess_conditions = [data['math score'] <= 10
    , (data['math score'] > 10) & (data['math score'] < 50)
    , (data['math score'] >= 50) & (data['math score'] < 70)
    , (data['math score'] >= 70) & (data['math score'] < 80)
    , (data['math score'] >= 80) & (data['math score'] <= 100)]
math_guess_values_min = [0,
                         data['math score'] - 10,
                         data['math score'] - 10,
                         data['math score'] - 10,
                         data['math score'] - 10
                         ]
math_guess_values_max = [data['math score'] + 25,
                         data['math score'] + 25,
                         round(data['math score'] + 20 + 0.05 * data['Extra study math percentage']),
                         data['math score'] + 20, 100]
data['math_guess_min'] = np.select(math_guess_conditions, math_guess_values_min)
data['math_guess_max'] = np.select(math_guess_conditions, math_guess_values_max)

# Reading
reading_guess_conditions = [data['reading score'] <= 10
    , (data['reading score'] > 10) & (data['reading score'] < 50)
    , (data['reading score'] >= 50) & (data['reading score'] < 70)
    , (data['reading score'] >= 70) & (data['reading score'] < 80)
    , (data['reading score'] >= 80) & (data['reading score'] <= 100)]
reading_guess_values_min = [0,
                            data['reading score'] - 10,
                            data['reading score'] - 10,
                            data['reading score'] - 10,
                            data['reading score'] - 10
                            ]
reading_guess_values_max = [data['reading score'] + 25,
                            data['reading score'] + 25,
                            round(data['reading score'] + 20 + 0.05 * data['Extra study reading percentage']),
                            data['reading score'] + 20, 100]
data['reading_guess_min'] = np.select(reading_guess_conditions, reading_guess_values_min)
data['reading_guess_max'] = np.select(reading_guess_conditions, reading_guess_values_max)

# Writing
writing_guess_conditions = [data['writing score'] <= 10
    , (data['writing score'] > 10) & (data['writing score'] < 50)
    , (data['writing score'] >= 50) & (data['writing score'] < 70)
    , (data['writing score'] >= 70) & (data['writing score'] < 80)
    , (data['writing score'] >= 80) & (data['writing score'] <= 100)]
writing_guess_values_min = [0,
                            data['writing score'] - 10,
                            data['writing score'] - 10,
                            data['writing score'] - 10,
                            data['writing score'] - 10
                            ]
writing_guess_values_max = [data['writing score'] + 25,
                            data['writing score'] + 25,
                            round(data['writing score'] + 20 + 0.05 * data['Extra study writing percentage']),
                            data['writing score'] + 20, 100]
data['writing_guess_min'] = np.select(writing_guess_conditions, writing_guess_values_min)
data['writing_guess_max'] = np.select(writing_guess_conditions, writing_guess_values_max)
data.to_csv('StudentResults_and_StudentResolves.csv')

# Print all result on Python
#######################################################################################

# Math
print("***Math***\n")
print(f"The average score of math : {data['math score'].mean()}")
t = data['math score'][data['math score'] >= 50].count()
z = data['math score'][data['math score'] >= 70].count()
print(f"math passed {t} Percent: {t / 10} %")
print(f"math not passed {1000 / t} Percent: {(1000 - t) / 10}%")
print(f"math extra study\n100%:", 1000 - t, "\n5-95%:", 1000 - (1000 - t) - z, "\n0%:", z)
t = data['math_guess_min'][data['math_guess_min'] >= 50].count()
z = data['math_guess_max'][data['math_guess_max'] >= 50].count()
print(f"Guessing math point range from min:\n>=50:{t}Percent:{t / 10}\n<50:{1000 - t}Percent:{(1000 - t) / 10}")
print(f"Guessing math point range from max:\n>=50:{z}Percent:{z / 10}\n<50:{1000 - z}Percent:{(1000 - z) / 10}")

# Reading
print("***Reading***\n")
print(f"The average score of reading : {data['reading score'].mean()}")
t = data['reading score'][data['reading score'] >= 50].count()
z = data['reading score'][data['reading score'] >= 70].count()
print(f"Reading passed {t} Percent: {t / 10} %")
print(f"Reading not passed {1000 / t} Percent: {(1000 - t) / 10}%")
print(f"Reading extra study\n100%:", 1000 - t, "\n5-95%:", 1000 - (1000 - t) - z, "\n0%:", z)
t = data['reading_guess_min'][data['reading_guess_min'] >= 50].count()
z = data['reading_guess_max'][data['reading_guess_max'] >= 50].count()
print(f"Guessing reading point range from min:\n>=50:{t}Percent:{t / 10}\n<50:{1000 - t}Percent:{(1000 - t) / 10}")
print(f"Guessing reading point range from max:\n>=50:{z}Percent:{z / 10}\n<50:{1000 - z}Percent:{(1000 - z) / 10}")

# Writing
print("***Writing***\n")
print(f"The average score of writing : {data['writing score'].mean()}")
t = data['writing score'][data['writing score'] >= 50].count()
z = data['writing score'][data['writing score'] >= 70].count()
print(f"writing passed {t} Percent: {t / 10} %")
print(f"writing not passed {1000 / t} Percent: {(1000 - t) / 10}%")
print(f"writing extra study\n100%:", 1000 - t, "\n5-95%:", 1000 - (1000 - t) - z, "\n0%:", z)
t = data['writing_guess_min'][data['writing_guess_min'] >= 50].count()
z = data['writing_guess_max'][data['writing_guess_max'] >= 50].count()
print(f"Guessing writing point range from min:\n>=50:{t}Percent:{t / 10}\n<50:{1000 - t}Percent:{(1000 - t) / 10}")
print(f"Guessing writing point range from max:\n>=50:{z}Percent:{z / 10}\n<50:{1000 - z}Percent:{(1000 - z) / 10}")

# Total tier
print("\n***Total tier in current result***\n")
print(f"Total passed:{data['Total tier'][data['Total tier'] == 'pass'].count()}")
print(f"Not passed:{data['Total tier'][data['Total tier'] == 'not pass'].count()}")

#######################################################################################
# End time of the process
end_time = time.time()
elapsed_time = end_time - start_time
print("elapsed_time:{0}".format(elapsed_time) + "[sec]")

#######################################################################################

plotbarscore = plt.figure("Math, Reading, Writing and Total score")

# Draw bar chart for Math


math_score_range = ["0", "1-10", "11-20", "21-30", "31-40", "41-50", "51-60", "61-70", "71-80", "81-90", "91-100"]
math_score_quantity = [
    data["math score"][(data["math score"] > ((i - 1) * 10)) & (data["math score"] <= (i * 10))].count() for i in
    range(11)]
plt.subplot(3, 1, 1)
plt.bar(math_score_range, math_score_quantity)
plt.title('Math score show')
plt.xlabel('Math Score')
plt.ylabel('Number of Student')

# Draw bar chart for Reading


reading_score_range = ["0", "1-10", "11-20", "21-30", "31-40", "41-50", "51-60", "61-70", "71-80", "81-90", "91-100"]
reading_score_quantity = [
    data["reading score"][(data["reading score"] > ((i - 1) * 10)) & (data["reading score"] <= (i * 10))].count() for i
    in range(11)]
plt.subplot(3, 1, 2)
plt.bar(reading_score_range, reading_score_quantity)
plt.title('Reading score show')
plt.xlabel('Reading Score')
plt.ylabel('Number of Student')

# Draw bar chart for Writing


writing_score_range = ["0", "1-10", "11-20", "21-30", "31-40", "41-50", "51-60", "61-70", "71-80", "81-90", "91-100"]
writing_score_quantity = [
    data["writing score"][(data["writing score"] > ((i - 1) * 10)) & (data["writing score"] <= (i * 10))].count() for i
    in range(11)]
plt.subplot(3, 1, 3)
plt.bar(writing_score_range, writing_score_quantity)
plt.title('Writing score show')
plt.xlabel('Writing Score')
plt.ylabel('Number of Student')

#######################################################################################
# Draw bar chart to evaluate the 4 factors race, parent level of education, lunch and test preparation course by calculating average, max, min and deviation of 3 subjects


# Race/ethnicity


pltrace = plt.figure("Math, Reading and Writing average, max,min and deviation results in race/ethnicity")
racetype = ["group A", "group B", "group C", "group D"]

# Math

math_by_race_average = [average(data["math score"][data["race/ethnicity"] == racetype[i]]) for i in range(4)]
# print(f"Math by race average by group A = {math_by_race_average[0]},group B = {math_by_race_average[1]},group C = {math_by_race_average[2]},group D = {math_by_race_average[3]}")
math_by_race_deviation = [deviation(data["math score"][data["race/ethnicity"] == racetype[i]]) for i in range(4)]
# print(f"Math by race deviation by group A = {math_by_race_deviation[0]},group B = {math_by_race_deviation[1]},group C = {math_by_race_deviation[2]},group D = {math_by_race_deviation[3]}")
math_by_race_max = [data["math score"][data["race/ethnicity"] == racetype[i]].max() for i in range(4)]
math_by_race_min = [data["math score"][data["race/ethnicity"] == racetype[i]].min() for i in range(4)]
plt.subplot(3, 1, 1)
math_x_axis_race = np.arange(len(racetype))
plt.bar(math_x_axis_race - 0.4, math_by_race_average, 0.2, label="Average")
plt.bar(math_x_axis_race - 0.2, math_by_race_deviation, 0.2, label="Deviation")
plt.bar(math_x_axis_race, math_by_race_max, 0.2, label="Max")
plt.bar(math_x_axis_race + 0.2, math_by_race_min, 0.2, label="Min")
plt.xticks(math_x_axis_race, racetype)
plt.title('Math score race description')
plt.xlabel('Math score race group')
plt.ylabel('Number of Student')
plt.legend()

# Reading

reading_by_race_average = [average(data["reading score"][data["race/ethnicity"] == racetype[i]]) for i in range(4)]
# print(f"reading by race average by group A = {reading_by_race_average[0]},group B = {reading_by_race_average[1]},group C = {reading_by_race_average[2]},group D = {reading_by_race_average[3]}")
reading_by_race_deviation = [deviation(data["reading score"][data["race/ethnicity"] == racetype[i]]) for i in range(4)]
# print(f"reading by race deviation by group A = {reading_by_race_deviation[0]},group B = {reading_by_race_deviation[1]},group C = {reading_by_race_deviation[2]},group D = {reading_by_race_deviation[3]}")
reading_by_race_max = [data["reading score"][data["race/ethnicity"] == racetype[i]].max() for i in range(4)]
reading_by_race_min = [data["reading score"][data["race/ethnicity"] == racetype[i]].min() for i in range(4)]
plt.subplot(3, 1, 2)
reading_x_axis_race = np.arange(len(racetype))
plt.bar(reading_x_axis_race - 0.4, reading_by_race_average, 0.2, label="Average")
plt.bar(reading_x_axis_race - 0.2, reading_by_race_deviation, 0.2, label="Deviation")
plt.bar(reading_x_axis_race, reading_by_race_max, 0.2, label="Max")
plt.bar(reading_x_axis_race + 0.2, reading_by_race_min, 0.2, label="Min")
plt.xticks(reading_x_axis_race, racetype)
plt.title('Reading score race description')
plt.xlabel('Reading score race group')
plt.ylabel('Number of Student')
plt.legend()

# Writing

writing_by_race_average = [average(data["writing score"][data["race/ethnicity"] == racetype[i]]) for i in range(4)]
# print(f"writing by race average by group A = {writing_by_race_average[0]},group B = {writing_by_race_average[1]},group C = {writing_by_race_average[2]},group D = {writing_by_race_average[3]}")
writing_by_race_deviation = [deviation(data["writing score"][data["race/ethnicity"] == racetype[i]]) for i in range(4)]
# print(f"writing by race deviation by group A = {writing_by_race_deviation[0]},group B = {writing_by_race_deviation[1]},group C = {writing_by_race_deviation[2]},group D = {writing_by_race_deviation[3]}")
writing_by_race_max = [data["writing score"][data["race/ethnicity"] == racetype[i]].max() for i in range(4)]
writing_by_race_min = [data["writing score"][data["race/ethnicity"] == racetype[i]].min() for i in range(4)]
plt.subplot(3, 1, 3)
writing_x_axis_race = np.arange(len(racetype))
plt.bar(writing_x_axis_race - 0.4, writing_by_race_average, 0.2, label="Average")
plt.bar(writing_x_axis_race - 0.2, writing_by_race_deviation, 0.2, label="Deviation")
plt.bar(writing_x_axis_race, writing_by_race_max, 0.2, label="Max")
plt.bar(writing_x_axis_race + 0.2, writing_by_race_min, 0.2, label="Min")
plt.xticks(writing_x_axis_race, racetype)
plt.title('Writing score race description')
plt.xlabel('Writing score race group')
plt.ylabel('Number of Student')
plt.legend()

#######################################################################################


# Parent level of education

pltparent_lv_edu = plt.figure(
    "Math, Reading and Writing average, max,min and deviation results in parent level of education")
parent_lv_edu_type = ["high school", "college", "associate's degree", "bachelor's degree", "master's degree"]

# Math

math_by_parent_lv_edu_average = [
    average(data["math score"][data["parental level of education"] == parent_lv_edu_type[i]]) for i in range(5)]
math_by_parent_lv_edu_deviation = [
    deviation(data["math score"][data["parental level of education"] == parent_lv_edu_type[i]]) for i in range(5)]
math_by_parent_lv_edu_max = [data["math score"][data["parental level of education"] == parent_lv_edu_type[i]].max() for
                             i in range(5)]
math_by_parent_lv_edu_min = [data["math score"][data["parental level of education"] == parent_lv_edu_type[i]].min() for
                             i in range(5)]
plt.subplot(3, 1, 1)
math_x_axis_parent_lv_edu = np.arange(len(parent_lv_edu_type))
plt.bar(math_x_axis_parent_lv_edu - 0.4, math_by_parent_lv_edu_average, 0.2, label="Average")
plt.bar(math_x_axis_parent_lv_edu - 0.2, math_by_parent_lv_edu_deviation, 0.2, label="Deviation")
plt.bar(math_x_axis_parent_lv_edu, math_by_parent_lv_edu_max, 0.2, label="Max")
plt.bar(math_x_axis_parent_lv_edu + 0.2, math_by_parent_lv_edu_min, 0.2, label="Min")
plt.xticks(math_x_axis_parent_lv_edu, parent_lv_edu_type)
plt.title('Math score parent_lv_edu description')
plt.xlabel('Math score parent_lv_edu group')
plt.ylabel('Number of Student')
plt.legend()

# Reading

reading_by_parent_lv_edu_average = [
    average(data["reading score"][data["parental level of education"] == parent_lv_edu_type[i]]) for i in range(5)]
reading_by_parent_lv_edu_deviation = [
    deviation(data["reading score"][data["parental level of education"] == parent_lv_edu_type[i]]) for i in range(5)]
reading_by_parent_lv_edu_max = [
    data["reading score"][data["parental level of education"] == parent_lv_edu_type[i]].max() for i in range(5)]
reading_by_parent_lv_edu_min = [
    data["reading score"][data["parental level of education"] == parent_lv_edu_type[i]].min() for i in range(5)]
plt.subplot(3, 1, 2)
reading_x_axis_parent_lv_edu = np.arange(len(parent_lv_edu_type))
plt.bar(reading_x_axis_parent_lv_edu - 0.4, reading_by_parent_lv_edu_average, 0.2, label="Average")
plt.bar(reading_x_axis_parent_lv_edu - 0.2, reading_by_parent_lv_edu_deviation, 0.2, label="Deviation")
plt.bar(reading_x_axis_parent_lv_edu, reading_by_parent_lv_edu_max, 0.2, label="Max")
plt.bar(reading_x_axis_parent_lv_edu + 0.2, reading_by_parent_lv_edu_min, 0.2, label="Min")
plt.xticks(reading_x_axis_parent_lv_edu, parent_lv_edu_type)
plt.title('Reading score parent_lv_edu description')
plt.xlabel('Reading score parent_lv_edu group')
plt.ylabel('Number of Student')
plt.legend()

# Writing

writing_by_parent_lv_edu_average = [
    average(data["writing score"][data["parental level of education"] == parent_lv_edu_type[i]]) for i in range(5)]
writing_by_parent_lv_edu_deviation = [
    deviation(data["writing score"][data["parental level of education"] == parent_lv_edu_type[i]]) for i in range(5)]
writing_by_parent_lv_edu_max = [
    data["writing score"][data["parental level of education"] == parent_lv_edu_type[i]].max() for i in range(5)]
writing_by_parent_lv_edu_min = [
    data["writing score"][data["parental level of education"] == parent_lv_edu_type[i]].min() for i in range(5)]
plt.subplot(3, 1, 3)
writing_x_axis_parent_lv_edu = np.arange(len(parent_lv_edu_type))
plt.bar(writing_x_axis_parent_lv_edu - 0.4, writing_by_parent_lv_edu_average, 0.2, label="Average")
plt.bar(writing_x_axis_parent_lv_edu - 0.2, writing_by_parent_lv_edu_deviation, 0.2, label="Deviation")
plt.bar(writing_x_axis_parent_lv_edu, writing_by_parent_lv_edu_max, 0.2, label="Max")
plt.bar(writing_x_axis_parent_lv_edu + 0.2, writing_by_parent_lv_edu_min, 0.2, label="Min")
plt.xticks(writing_x_axis_parent_lv_edu, parent_lv_edu_type)
plt.title('writing score parent_lv_edu description')
plt.xlabel('writing score parent_lv_edu group')
plt.ylabel('Number of Student')
plt.legend()
plt.tight_layout()

#######################################################################################



# Lunch


pltlunch = plt.figure("Math, Reading and Writing average, max,min and deviation results in lunch")
lunch_type = ["standard", "free/reduced"]

# Math

math_by_lunch_average = [average(data["math score"][data["lunch"] == lunch_type[i]]) for i in range(2)]
math_by_lunch_deviation = [deviation(data["math score"][data["lunch"] == lunch_type[i]]) for i in range(2)]
math_by_lunch_max = [data["math score"][data["parental level of education"] == lunch_type[i]].max() for i in range(2)]
math_by_lunch_min = [data["math score"][data["parental level of education"] == lunch_type[i]].min() for i in range(2)]
plt.subplot(3, 1, 1)
math_x_axis_lunch = np.arange(len(lunch_type))
plt.bar(math_x_axis_lunch - 0.4, math_by_lunch_average, 0.2, label="Average")
plt.bar(math_x_axis_lunch - 0.2, math_by_lunch_deviation, 0.2, label="Deviation")
plt.bar(math_x_axis_lunch, math_by_lunch_max, 0.2, label="Max")
plt.bar(math_x_axis_lunch + 0.2, math_by_lunch_min, 0.2, label="Min")
plt.xticks(math_x_axis_lunch, lunch_type)
plt.title('Math score lunch  description')
plt.xlabel('Math score lunch  group')
plt.ylabel('Number of Student')
plt.legend()

# Reading

reading_by_lunch_average = [average(data["reading score"][data["lunch"] == lunch_type[i]]) for i in range(2)]
reading_by_lunch_deviation = [deviation(data["reading score"][data["lunch"] == lunch_type[i]]) for i in range(2)]
reading_by_lunch_max = [data["reading score"][data["parental level of education"] == lunch_type[i]].max() for i in
                        range(2)]
reading_by_lunch_min = [data["reading score"][data["parental level of education"] == lunch_type[i]].min() for i in
                        range(2)]
plt.subplot(3, 1, 2)
reading_x_axis_lunch = np.arange(len(lunch_type))
plt.bar(reading_x_axis_lunch - 0.4, reading_by_lunch_average, 0.2, label="Average")
plt.bar(reading_x_axis_lunch - 0.2, reading_by_lunch_deviation, 0.2, label="Deviation")
plt.bar(reading_x_axis_lunch, reading_by_lunch_max, 0.2, label="Max")
plt.bar(reading_x_axis_lunch + 0.2, reading_by_lunch_min, 0.2, label="Min")
plt.xticks(reading_x_axis_lunch, lunch_type)
plt.title('reading score lunch  description')
plt.xlabel('reading score lunch  group')
plt.ylabel('Number of Student')
plt.legend()

# Writing

writing_by_lunch_average = [average(data["writing score"][data["lunch"] == lunch_type[i]]) for i in range(2)]
writing_by_lunch_deviation = [deviation(data["writing score"][data["lunch"] == lunch_type[i]]) for i in range(2)]
writing_by_lunch_max = [data["writing score"][data["parental level of education"] == lunch_type[i]].max() for i in
                        range(2)]
writing_by_lunch_min = [data["writing score"][data["parental level of education"] == lunch_type[i]].min() for i in
                        range(2)]
plt.subplot(3, 1, 3)
writing_x_axis_lunch = np.arange(len(lunch_type))
plt.bar(writing_x_axis_lunch - 0.4, writing_by_lunch_average, 0.2, label="Average")
plt.bar(writing_x_axis_lunch - 0.2, writing_by_lunch_deviation, 0.2, label="Deviation")
plt.bar(writing_x_axis_lunch, writing_by_lunch_max, 0.2, label="Max")
plt.bar(writing_x_axis_lunch + 0.2, writing_by_lunch_min, 0.2, label="Min")
plt.xticks(writing_x_axis_lunch, lunch_type)
plt.title('writing score lunch  description')
plt.xlabel('writing score lunch  group')
plt.ylabel('Number of Student')
plt.legend()
plt.tight_layout()

#######################################################################################


plotpie = plt.figure("Math, Reading, Writing and Total evaluation")
chartLabels = ['Passed', 'Failed']

# Draw pie chart for Math

passMathCount = data['Math tier'][data['Math tier'] == 'pass'].count()
failedMathCount = data['Math tier'][data['Math tier'] == 'fail'].count()
chartContent = [passMathCount, failedMathCount]
plt.subplot(2, 2, 1)  # Figure will have 2 rows and 2 column, this plot is at position 1
plt.pie(chartContent, labels=chartLabels, startangle=90, autopct='%1.1f%%')
plt.legend()
plt.title("Students Math Result")

# Draw pie chart for Reading

passReadCount = data['Reading tier'][data['Reading tier'] == 'pass'].count()
failedReadCount = data['Reading tier'][data['Reading tier'] == 'fail'].count()
chartContent = [passReadCount, failedReadCount]
plt.subplot(2, 2, 2)  # Figure will have 2 rows and 2 column, this plot is at position 2
plt.pie(chartContent, labels=chartLabels, startangle=90, autopct='%1.1f%%')
plt.legend()
plt.title("Students Reading Result")

# Draw pie chart for Writing

passWriteCount = data['Writing tier'][data['Writing tier'] == 'pass'].count()
failedWriteCount = data['Writing tier'][data['Writing tier'] == 'fail'].count()
chartContent = [passWriteCount, failedWriteCount]
plt.subplot(2, 2, 3)  # Figure will have 2 rows and 2 column, this plot is at position 3
plt.pie(chartContent, labels=chartLabels, startangle=90, autopct='%1.1f%%')
plt.legend()
plt.title("Students Writing Result")

# Draw pie chart for Overall Result

PassedCount = data['Total tier'][data['Total tier'] == 'pass'].count()
FailedCount = data['Total tier'][data['Total tier'] == 'fail'].count()
chartContent = [PassedCount, FailedCount]
plt.subplot(2, 2, 4)  # Figure will have 2 rows and 2 column, this plot is at position 4
plt.pie(chartContent, labels=chartLabels, startangle=90, autopct='%1.1f%%')
plt.legend()
plt.title("Students Overall Result")
plt.suptitle("STUDENTS EXAM PERFORMANCE")
# plt.tight_layout()

#######################################################################################


# Lunch


pltlunch = plt.figure("Math, Reading and Writing average, max,min and deviation results in lunch")
lunch_type = ["standard", "free/reduced"]

# Math

math_by_lunch_average = [average(data["math score"][data["lunch"] == lunch_type[i]]) for i in range(2)]
math_by_lunch_deviation = [deviation(data["math score"][data["lunch"] == lunch_type[i]]) for i in range(2)]
math_by_lunch_max = [data["math score"][data["lunch"] == lunch_type[i]].max() for i in range(2)]
math_by_lunch_min = [data["math score"][data["lunch"] == lunch_type[i]].min() for i in range(2)]
plt.subplot(3, 1, 1)
math_x_axis_lunch = np.arange(len(lunch_type))
plt.bar(math_x_axis_lunch - 0.4, math_by_lunch_average, 0.2, label="Average")
plt.bar(math_x_axis_lunch - 0.2, math_by_lunch_deviation, 0.2, label="Deviation")
plt.bar(math_x_axis_lunch, math_by_lunch_max, 0.2, label="Max")
plt.bar(math_x_axis_lunch + 0.2, math_by_lunch_min, 0.2, label="Min")
plt.xticks(math_x_axis_lunch, lunch_type)
plt.title('Math score lunch description')
plt.xlabel('Math score lunch group')
plt.ylabel('Number of Student')
plt.legend()

# Reading

reading_by_lunch_average = [average(data["reading score"][data["lunch"] == lunch_type[i]]) for i in range(2)]
reading_by_lunch_deviation = [deviation(data["reading score"][data["lunch"] == lunch_type[i]]) for i in range(2)]
reading_by_lunch_max = [data["reading score"][data["lunch"] == lunch_type[i]].max() for i in range(2)]
reading_by_lunch_min = [data["reading score"][data["lunch"] == lunch_type[i]].min() for i in range(2)]
plt.subplot(3, 1, 2)
reading_x_axis_lunch = np.arange(len(lunch_type))
plt.bar(reading_x_axis_lunch - 0.4, reading_by_lunch_average, 0.2, label="Average")
plt.bar(reading_x_axis_lunch - 0.2, reading_by_lunch_deviation, 0.2, label="Deviation")
plt.bar(reading_x_axis_lunch, reading_by_lunch_max, 0.2, label="Max")
plt.bar(reading_x_axis_lunch + 0.2, reading_by_lunch_min, 0.2, label="Min")
plt.xticks(reading_x_axis_lunch, lunch_type)
plt.title('reading score lunch description')
plt.xlabel('reading score lunch group')
plt.ylabel('Number of Student')
plt.legend()

# Writing

writing_by_lunch_average = [average(data["writing score"][data["lunch"] == lunch_type[i]]) for i in range(2)]
writing_by_lunch_deviation = [deviation(data["writing score"][data["lunch"] == lunch_type[i]]) for i in range(2)]
writing_by_lunch_max = [data["writing score"][data["lunch"] == lunch_type[i]].max() for i in range(2)]
writing_by_lunch_min = [data["writing score"][data["lunch"] == lunch_type[i]].min() for i in range(2)]
plt.subplot(3, 1, 3)
writing_x_axis_lunch = np.arange(len(lunch_type))
plt.bar(writing_x_axis_lunch - 0.4, writing_by_lunch_average, 0.2, label="Average")
plt.bar(writing_x_axis_lunch - 0.2, writing_by_lunch_deviation, 0.2, label="Deviation")
plt.bar(writing_x_axis_lunch, writing_by_lunch_max, 0.2, label="Max")
plt.bar(writing_x_axis_lunch + 0.2, writing_by_lunch_min, 0.2, label="Min")
plt.xticks(writing_x_axis_lunch, lunch_type)
plt.title('writing score lunch description')
plt.xlabel('writing score lunch group')
plt.ylabel('Number of Student')
plt.legend()
# plt.tight_layout()


#######################################################################################


# Test_preparation


plttest_prepartion = plt.figure("Math, Reading and Writing average, max,min and deviation results in Test_preparation")
test_prepartion_type = ["none", "completed"]

# Math

math_by_test_preparation_average = [
    average(data["math score"][data["test preparation course"] == test_prepartion_type[i]]) for i in range(2)]
math_by_test_preparation_deviation = [
    deviation(data["math score"][data["test preparation course"] == test_prepartion_type[i]]) for i in range(2)]
math_by_test_preparation_max = [data["math score"][data["test preparation course"] == test_prepartion_type[i]].max() for
                                i in range(2)]
math_by_test_preparation_min = [data["math score"][data["test preparation course"] == test_prepartion_type[i]].min() for
                                i in range(2)]
plt.subplot(3, 1, 1)
math_x_axis_test_prepartion = np.arange(len(test_prepartion_type))
plt.bar(math_x_axis_test_prepartion - 0.4, math_by_test_preparation_average, 0.2, label="Average")
plt.bar(math_x_axis_test_prepartion - 0.2, math_by_test_preparation_deviation, 0.2, label="Deviation")
plt.bar(math_x_axis_test_prepartion, math_by_test_preparation_max, 0.2, label="Max")
plt.bar(math_x_axis_test_prepartion + 0.2, math_by_test_preparation_min, 0.2, label="Min")
plt.xticks(math_x_axis_test_prepartion, test_prepartion_type)
plt.title('Math score test preparation description')
plt.xlabel('Math score test preparation group')
plt.ylabel('Number of Student')
plt.legend()

# Reading

reading_by_test_preparation_average = [
    average(data["reading score"][data["test preparation course"] == test_prepartion_type[i]]) for i in range(2)]
reading_by_test_preparation_deviation = [
    deviation(data["reading score"][data["test preparation course"] == test_prepartion_type[i]]) for i in range(2)]
reading_by_test_preparation_max = [
    data["reading score"][data["test preparation course"] == test_prepartion_type[i]].max() for i in range(2)]
reading_by_test_preparation_min = [
    data["reading score"][data["test preparation course"] == test_prepartion_type[i]].min() for i in range(2)]
plt.subplot(3, 1, 2)
reading_x_axis_test_prepartion = np.arange(len(test_prepartion_type))
plt.bar(reading_x_axis_test_prepartion - 0.4, reading_by_test_preparation_average, 0.2, label="Average")
plt.bar(reading_x_axis_test_prepartion - 0.2, reading_by_test_preparation_deviation, 0.2, label="Deviation")
plt.bar(reading_x_axis_test_prepartion, reading_by_test_preparation_max, 0.2, label="Max")
plt.bar(reading_x_axis_test_prepartion + 0.2, reading_by_test_preparation_min, 0.2, label="Min")
plt.xticks(reading_x_axis_test_prepartion, test_prepartion_type)
plt.title('reading score test preparation description')
plt.xlabel('reading score test preparation group')
plt.ylabel('Number of Student')
plt.legend()

# Writing

writing_by_test_preparation_average = [
    average(data["writing score"][data["test preparation course"] == test_prepartion_type[i]]) for i in range(2)]
writing_by_test_preparation_deviation = [
    deviation(data["writing score"][data["test preparation course"] == test_prepartion_type[i]]) for i in range(2)]
writing_by_test_preparation_max = [
    data["writing score"][data["test preparation course"] == test_prepartion_type[i]].max() for i in range(2)]
writing_by_test_preparation_min = [
    data["writing score"][data["test preparation course"] == test_prepartion_type[i]].min() for i in range(2)]
plt.subplot(3, 1, 3)
writing_x_axis_test_prepartion = np.arange(len(test_prepartion_type))
plt.bar(writing_x_axis_test_prepartion - 0.4, writing_by_test_preparation_average, 0.2, label="Average")
plt.bar(writing_x_axis_test_prepartion - 0.2, writing_by_test_preparation_deviation, 0.2, label="Deviation")
plt.bar(writing_x_axis_test_prepartion, writing_by_test_preparation_max, 0.2, label="Max")
plt.bar(writing_x_axis_test_prepartion + 0.2, writing_by_test_preparation_min, 0.2, label="Min")
plt.xticks(writing_x_axis_test_prepartion, test_prepartion_type)
plt.title('writing score test preparation description')
plt.xlabel('writing score test preparation group')
plt.ylabel('Number of Student')
plt.legend()
plt.tight_layout()

#######################################################################################
# Math, Reading, Writing Tutorial time learn percentage


plotbartutorial = plt.figure("Math, Reading, Writing Tutorial time learn percentage")
# Math

mathper, readingper, writingper = [5] * 6, [5] * 6, [5] * 6
for i in range(6):
    mathper[i] = data["Extra study math percentage"][(data["Extra study math percentage"] > ((i - 1) * 25)) & (
                data["Extra study math percentage"] <= (i * 25))].count()
    if i == 5:
        mathper[i] = data["Extra study math percentage"][data["Extra study math percentage"] == 100].count()
mathpernum = [mathper[i] for i in range(6)]
mathper = ["0%", "0-25%", "25-50%", "50-75%", "75-100%", "100%"]
plt.subplot(3, 1, 1)
plt.bar(mathper, mathpernum)
plt.title('Math tutorial study percentage')
plt.xlabel('Percent')
plt.ylabel('Number of Student')

# Reading

for i in range(6):
    readingper[i] = data["Extra study reading percentage"][(data["Extra study reading percentage"] > ((i - 1) * 25)) & (
                data["Extra study reading percentage"] <= (i * 25))].count()
    if i == 5:
        readingper[i] = data["Extra study reading percentage"][data["Extra study reading percentage"] == 100].count()
readingpernum = [readingper[i] for i in range(6)]
readingper = ["0%", "0-25%", "25-50%", "50-75%", "75-100%", "100%"]
plt.subplot(3, 1, 2)
plt.bar(readingper, readingpernum)
plt.title('Reading tutorial study percentage')
plt.xlabel('Percent')
plt.ylabel('Number of Student')

# writing

for i in range(6):
    writingper[i] = data["Extra study writing percentage"][(data["Extra study writing percentage"] > ((i - 1) * 25)) & (
                data["Extra study writing percentage"] <= (i * 25))].count()
    if i == 5:
        writingper[i] = data["Extra study writing percentage"][data["Extra study writing percentage"] == 100].count()
writingpernum = [writingper[i] for i in range(6)]
writingper = ["0%", "0-25%", "25-50%", "50-75%", "75-100%", "100%"]
plt.subplot(3, 1, 3)
plt.bar(writingper, writingpernum)
plt.title('writing tutorial study percentage')
plt.xlabel('Percent')
plt.ylabel('Number of Student')
plt.tight_layout()

#######################################################################################
# Plot the guess point from min to max


plotguesspoint = plt.figure("Point guessing")

# Math

mathguessmin = [
    data["math_guess_min"][(data["math_guess_min"] > ((i - 1) * 10)) & (data["math_guess_min"] <= (i * 10))].count() for
    i in range(11)]
mathguessmax = [
    data["math_guess_max"][(data["math_guess_max"] > ((i - 1) * 10)) & (data["math_guess_max"] <= (i * 10))].count() for
    i in range(11)]
math_score_range = ["0", "1-10", "11-20", "21-30", "31-40", "41-50", "51-60", "61-70", "71-80", "81-90", "91-100"]
plt.subplot(3, 1, 1)
math_x_axis = np.arange(len(math_score_range))
plt.bar(math_x_axis - 0.2, mathguessmin, 0.4, label="Min")
plt.bar(math_x_axis + 0.2, mathguessmax, 0.4, label="Max")
plt.xticks(math_x_axis, math_score_range)
plt.title('Math score guessing show')
plt.xlabel('Math Score guessing')
plt.ylabel('Number of Student')
plt.legend()

# Reading

readingguessmin = [data["reading_guess_min"][
                       (data["reading_guess_min"] > ((i - 1) * 10)) & (data["reading_guess_min"] <= (i * 10))].count()
                   for i in range(11)]
readingguessmax = [data["reading_guess_max"][
                       (data["reading_guess_max"] > ((i - 1) * 10)) & (data["reading_guess_max"] <= (i * 10))].count()
                   for i in range(11)]
reading_score_range = ["0", "1-10", "11-20", "21-30", "31-40", "41-50", "51-60", "61-70", "71-80", "81-90", "91-100"]
plt.subplot(3, 1, 2)
reading_x_axis = np.arange(len(reading_score_range))
plt.bar(reading_x_axis - 0.2, readingguessmin, 0.4, label="Min")
plt.bar(reading_x_axis + 0.2, readingguessmax, 0.4, label="Max")
plt.xticks(reading_x_axis, reading_score_range)
plt.title('Reading score guessing show')
plt.xlabel('Reading Score guessing')
plt.ylabel('Number of Student')
plt.legend()

# Writing

writingguessmin = [data["writing_guess_min"][
                       (data["writing_guess_min"] > ((i - 1) * 10)) & (data["writing_guess_min"] <= (i * 10))].count()
                   for i in range(11)]
writingguessmax = [data["writing_guess_max"][
                       (data["writing_guess_max"] > ((i - 1) * 10)) & (data["writing_guess_max"] <= (i * 10))].count()
                   for i in range(11)]
writing_score_range = ["0", "1-10", "11-20", "21-30", "31-40", "41-50", "51-60", "61-70", "71-80", "81-90", "91-100"]
plt.subplot(3, 1, 3)
writing_x_axis = np.arange(len(writing_score_range))
plt.bar(writing_x_axis - 0.2, writingguessmin, 0.4, label="Min")
plt.bar(writing_x_axis + 0.2, writingguessmax, 0.4, label="Max")
plt.xticks(writing_x_axis, writing_score_range)
plt.title('Writing score guessing show')
plt.xlabel('Writing Score guessing')
plt.ylabel('Number of Student')
plt.legend()

#######################################################################################

# Show all the figure

plt.tight_layout()
plt.show()

#######################################################################################
# Time consuming- Might not be needed
