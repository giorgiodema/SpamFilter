from functools import reduce


        


class BayesianClassifier:



    def __init__(self,tv,a,d):
        
        self.targetValues = tv      # [spam,ham]
        self.attributes = a         # [list of words appearing in all messages]
        self.dataset = d            # ["spam/ham", "word1", "word2", ..., "wordn"]

        self.priorProb = {}         # {"spam":'prior spam probability', "ham": 'prior ham probability'}
        self.likelihood = {}        # { ("spam/ham":"word1"):prob, ...}

        return


    def estimatePriorProb(self,tv):
        return len(list(filter(lambda x: x[0] == tv, self.dataset))) / len(self.dataset)


    def estimateLikelihood(self,tv,a):
        return len(list(filter(lambda x: x[0] == tv and a in x, self.dataset))) / len(list(filter(lambda x: x[0] == tv, self.dataset)))


    def Learn(self):

        for tv in self.targetValues:
            self.priorProb[tv] = self.estimatePriorProb(tv)

            for a in self.attributes:
                self.likelihood[(tv,a)] = self.estimateLikelihood(tv,a)


        return





    def Classify(self,document):

        posteriorProb = {}

        for tv in self.targetValues:
            priorElem = [x[1] for x in list(self.likelihood.items()) if x[0][0]==tv if x[0][1] in document]
            posteriorProb[tv] = 0 if len(priorElem)==0 else reduce(lambda x,y: x*y, priorElem) * self.priorProb[tv]

        inversePosteriorProb =  {v: k for k, v in posteriorProb.items()}
        max =  reduce(lambda x,y: x if(x>y) else y, inversePosteriorProb)

        return inversePosteriorProb[max]




