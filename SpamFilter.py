
"""

        targetValues = [spam,ham]
        
        attributes = [list of words appearing in all messages]

        dataset = [list of documents]

        document = { class:"HAM/SPAM", word1:freq1, word2:freq2, ... }

        aPriori = { "SPAM":aPrioriProb, "HAM":aprioriProb }
        likelihood = { ("SPAM":"word1"):prob, ...}




"""



class BayesianLearn:



    def __init__(self, targetValues, attributes, dataset):
        
        self.targetValues = targetValues
        self.attributes = attributes
        self.dataset = dataset

        self.priorProb = {}
        self.likelihood = {}

        return




    def Learn():

        for tv in targetValues:
            priorProb[tv] = estimatePriorProb(dataset,tv)

            for a in attributes:
                likelihood[(tv,a)] = estimateLikelihood(dataset,tv,a)


        return



    def Classify(document):

        posteriorProb = {}

        for tv in targetValues:
            posteriorProb[tv] = ## TODO








