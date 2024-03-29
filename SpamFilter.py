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
        raise NotImplementedError


    def estimateLikelihood(self,tv,a):
        raise NotImplementedError


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



class BinomialSpamFilter (BayesianClassifier):

    def __init__(self,tv,a,d):
        super().__init__(tv,a,d)


    def estimatePriorProb(self,tv):
        return len(list(filter(lambda x: x[0] == tv, self.dataset))) / len(self.dataset)


    def estimateLikelihood(self,tv,a):
        return len(list(filter(lambda x: x[0] == tv and a in x, self.dataset))) / len(list(filter(lambda x: x[0] == tv, self.dataset)))


class MultinomialSpamFilter (BayesianClassifier):

    def __init__(self,tv,a,d, alpha):                           # dataset = {"CLASS":spam/ham, "word1":freq1, ..., "wordn":freqn}
        super().__init__(tv,a,d)                                # NB: 'CLASS' must be uppercase to be dietingished from other words that are all lowercase
        self.alpha = alpha





    def estimatePriorProb(self,tv):
        return  len(list(filter(lambda x: x["CLASS"]==tv, self.dataset)))  /  len(self.dataset)



    def estimateLikelihood(self,tv,a):
        instancesClassTv = list(filter(lambda x: x["CLASS"]== tv, self.dataset))                     # documents of class tv
        instancesWithAttrAandClassTv = list(filter(lambda x: a in x, instancesClassTv))              # documents of class 'tv' containing word 'a'
    
        occAttrA = 0;                                                                                # occurrences of word 'a' in all documents of class 'tv' containing 'a'
        for d in instancesWithAttrAandClassTv:
            if a in d:
                occAttrA += d[a]

        occAllAttr = 0;                                                                              # all words frequencies in all documents of class 'tv'
        for d in instancesClassTv:                                                                   
            for a in d:
                if a!='CLASS':
                    occAllAttr += d[a]

                    

        return (occAttrA + self.alpha) / (occAllAttr + self.alpha*len(self.dataset))
