
import SpamFilter
import functools
import re

TRAINING_SET_SIZE = 0.2

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
numTries = len(testSet)
numSuccesses = 0

for i in range(0,len(testSet)):
    predictedValue = sf.Classify(testSet[i][1:])
    print("Test #{}    | Predicted: {}    | Expected: {}".format(i,predictedValue,testSet[i][0]))
    
    if predictedValue == testSet[i][0]:
        numSuccesses +=1

print("\n\nSuccess percentage: {} %".format((numSuccesses/numTries)*100))