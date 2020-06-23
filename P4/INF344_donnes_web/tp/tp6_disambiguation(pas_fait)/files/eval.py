# /usr/bin/python3

from sys import argv

if len(argv) != 4:
    print("Usage : evaluation.py goldstandard student debug")
    exit()

goldstandard_file = argv[1]
student_file = argv[2]
if argv[3] == "1":
    debug = True
else:
    debug = False

# Dictionaries
goldstandard = dict()
student = dict()

# Reading first file
with open(goldstandard_file) as f:
    for line in f:
        temp = line.split("\t")
        if len(temp) != 2:
            print("The line:", line, "has an incorrect number of tabs")
        else:
            if temp[0] in goldstandard:
                print(temp[0], " has two solutions")
            goldstandard[temp[0]] = temp[1]

# Reading second file
with open(student_file) as f:
    for line in f:
        temp = line.split("\t")
        if len(temp) != 2:
            if not debug:
                print("Comment :=>> The line: '", line, "' has an incorrect number of tabs")
            else:
                print("The line: '", line, "' has an incorrect number of tabs")
        else:
            if temp[0] in student:
                if not debug:
                    print("Comment :=>>", temp[0], "has two solutions")
                else:
                    print(temp[0], " has two solutions")
            student[temp[0]] = temp[1]

true_pos = 0
false_pos = 0
false_neg = 0

for key in student:
    if key in goldstandard:
        if student[key] == goldstandard[key]:
            true_pos += 1
        else:
            false_pos += 1
            if debug:
                print("You got", key, "wrong. Expected output:", goldstandard[key], "given", student[key])

for key in goldstandard:
    if key not in student:
        false_neg += 1
        if debug and false_neg < 50:
            print("No solution was given for", key)
        elif debug and false_neg == 50:
            print("Other solutions not found...")

if true_pos + false_pos != 0:
    precision = float(true_pos) / (true_pos + false_pos) * 100.0
else:
    precision = 0.0

if true_pos + false_neg != 0:
    recall = float(true_pos) / (true_pos + false_neg + false_pos) * 100.0
else:
    recall = 0.0

def fmeasure(beta, precision, recall):
    if precision + recall != 0.0:
        return (1.0 + beta * beta) * precision * recall / (beta * beta * precision + recall)
    else:
        return 0.0

f1 = fmeasure(1, precision, recall)
#grade = 0.75 * precision + 0.25 * recall
grade = fmeasure(0.5, precision, recall)

if debug:
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", f1)
    print("Simulated Grade:", grade)
else:
    print("Comment :=>>", "Precision:", precision)
    print("Comment :=>>", "Recall:", recall)
    print("Comment :=>>", "F1:", f1)
    print("Comment :=>>", "Grade:", grade)
    print("Grade :=>>", grade)

