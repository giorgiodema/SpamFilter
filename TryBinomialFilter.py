
import SpamFilter
import functools
import re

TRAINING_SET_SIZE = 0.66

# LOADING DATA FROM FILE -------------------------------------------------------------
file = open("SMSSpamCollection","r").readlines()
regexp = re.compile("\W*")

dataset = list(map(lambda x: x[0:-1],list(map( lambda x: regexp.split(x),file))))
for i in range(0,len(dataset)):
    dataset[i] = list(map(lambda x:x.lower(),dataset[i]))


# SPLITTING DATASET INTO TEST AND TRAINING SET -----------------------------------------
trainingSet = dataset[0: int(len(dataset)*TRAINING_SET_SIZE)]
testSet     = dataset[int(len(dataset)*TRAINING_SET_SIZE):]



# LISTING TARGET VALUES AND ATTRIBUTES ----------------------------------------------
tv = ["ham","spam"]
attributes = []

for i in range(0,len(trainingSet)):
    attributes.extend([x for x in dataset[i] if not x in tv if not x in attributes])


# TRAINING SPAM FILTER --------------------------------------------------------------
sf = SpamFilter.BinomialSpamFilter(tv,attributes,trainingSet)
sf.Learn()



# TESTING SPAM FILTER --------------------------------------------------------------
numTotalTries = len(testSet)
numTotalSuccesses = 0

falsePositives = 0              # when a ham is classified as spam
falseNegatives = 0              # when spam is classified as ham


realPositives = 0              # number of spam
realNegatives = 0              # number of ham

confusionMatrix = [[0,0],[0,0]] # rows are the real classes 0->HAM 1->SPAM
                                # columns are the stimated classes 0->HAM 1->SPAM

for i in range(len(trainingSet),len(dataset)):
    predictedValue = sf.Classify(dataset[i][1:])
    print("Test #{}    | Predicted: {}    | Expected: {}".format(i,predictedValue,dataset[i][0]))
    
    if dataset[i][0] == "ham":
        realNegatives +=1
        if predictedValue == "ham":
            numTotalSuccesses +=1
            confusionMatrix[0][0] +=1
        else:
            falsePositives +=1
            confusionMatrix[0][1] +=1
    else:
        realPositives +=1
        if predictedValue =="spam":
            numTotalSuccesses +=1
            confusionMatrix[1][1] +=1
        else:
            falseNegatives +=1
            confusionMatrix[1][0] +=1


accuracy = numTotalSuccesses/numTotalTries
recall = realPositives/(realPositives + falseNegatives)     # ability to avoid false negatives (false ham)
precision = realPositives/(realPositives + falsePositives)  # ability to avoid false positives (false spam)


print("\n\nAccuracy: {} %".format(accuracy))
print("\n\nRecall    (ability to avoid to classify spam as ham): {} ".format(recall))
print("\n\nPrecision (ability to avoid to classify ham as spam): {}".format(precision))

print("\n\nCONFUSION MATRIX")
print("\nrows represent real classes (0->HAM 1->SPAM) ")
print("\ncols represent predicted classes (0->HAM 1->SPAM)\n\n")

for i in range(0,2):
    for j in range(0,2):
        print("{}".format(confusionMatrix[i][j]), end=" ")
    print("")
print("\n\n")