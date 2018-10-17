

import SpamFilter
import functools
import re

TRAINING_SET_SIZE = 0.8

# LOADING DATA FROM FILE -------------------------------------------------------------
file = open("SMSSpamCollection","r").readlines()
regexp = re.compile("\W*")

documents = list(map(lambda x: x[0:-1],list(map( lambda x: regexp.split(x),file))))
for i in range(0,len(documents)):
    documents[i] = list(map(lambda x:x.lower(),documents[i]))

dataset = []
for i in range(0, len(documents)):
    d = {}
    d["CLASS"] = documents[i][0]
    for x in documents[i][1:]:
        if x not in d:
            d[x] = 1
        else:
            d[x] = d[x]+1
    dataset.append(d)



# SPLITTING DATASET INTO TEST AND TRAINING SET -----------------------------------------
trainingSet = dataset[0: int(len(dataset)*TRAINING_SET_SIZE)]
testSet     = dataset[int(len(dataset)*TRAINING_SET_SIZE):]



# LISTING TARGET VALUES AND ATTRIBUTES ----------------------------------------------
tv = ["ham","spam"]
attributes = []

for i in range(0,len(trainingSet)):
    attributes.extend([x for x in dataset[i] if x!= 'CLASS' if not x in tv if not x in attributes])


# TRAINING SPAM FILTER --------------------------------------------------------------
sf = SpamFilter.MultinomialSpamFilter(tv,attributes,trainingSet,1)
sf.Learn()



# TESTING SPAM FILTER --------------------------------------------------------------
numTries = len(testSet)
numSuccesses = 0

for i in range(len(trainingSet),len(documents)):
    predictedValue = sf.Classify(documents[i][1:])
    print("Test #{}    | Predicted: {}    | Expected: {}".format(i,predictedValue,documents[i][0]))
    
    if predictedValue == documents[i][0]:
        numSuccesses +=1

print("\n\nSuccess percentage: {} %".format((numSuccesses/numTries)*100))