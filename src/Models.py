import numpy as np
import Tools
import nltk
from nltk.corpus import stopwords
import math
import re

# implement CBOW, Skip-Gram
class word2vec:

    def __init__(self, model, dimension, C, filename = '', hs = False, min_count = 0, stem = False):
        self.W_i = np.empty(0)
        self.W_o = np.empty(0)
        self.N = dimension
        self.C = C
        self.voc = Tools.Vocabulary()
        self.sentences = []
        self.hs = hs
        self.freq = {}
        self.ht = None
        self.model = 0 # 0: cbow, 1: skip-gram
        self.min_count = min_count
        self.stem = stem
        if model is 'cbow':
            self.model = 0
        elif model is 'skipgram':
            self.model = 1
        else:
            print("<!> model should be either \'cbow\' or \'skipgram\'")
            raise ValueError

        if(filename):
            self.inputData(filename)

    def inputData(self, filename): # preprocessing
        inputstr = ""
        words = []

        stpwrds = set(stopwords.words('english'))
        if self.stem:
            stemmer = nltk.stem.porter.PorterStemmer()
        f = open(filename)
        lines = f.readlines()
        for line in lines:
            line = line.lower()
            for sentence in re.split('\.|\!|\?|;|:', line):
                tmp = nltk.word_tokenize(sentence)
                tokens = []
                for token in tmp:
                    if token.isalpha() and token not in stpwrds:
                        if self.stem:
                            token = stemmer.stem(token)
                        tokens.append(token)
                        words.append(token)
                if len(tokens) > 1:
                    self.sentences.append(tokens)
                    if self.min_count == 0:
                        self.voc.add(tokens)

        self.freq = Tools.tokFreq(words)

        if self.min_count > 0:
            tmp = self.sentences
            self.sentences = []
            for sentence in tmp:
                sentence = [w for w in sentence if self.freq[w] >= self.min_count]
                if len(sentence) > 1:
                    self.voc.add(sentence)
                    self.sentences.append(sentence)
            words = [w for w in words if self.freq[w] >= self.min_count]
            self.freq = Tools.tokFreq(words)

        if self.hs:
            self.ht = Tools.HuffmanTree(self.N, self.freq)
        f.close()
        self.initWeights()
        print(self.freq)
        print(self.sentences)

    def initWeights(self):
        V = self.voc.size()
        self.W_i = np.random.rand(V, self.N)
        if not self.hs:
            self.W_o = np.random.rand(self.N, V)

    def train(self, learning_rate):
        if self.model:
            self.train_skipgram(learning_rate)
        else:
            self.train_cbow(learning_rate)
        
    def train_skipgram(self, learning_rate):
        for sentence in self.sentences:
            for i in range(len(sentence)):

                context_start = max(0, int(i - self.C / 2))
                context_end = min(len(sentence), int(i + self.C / 2 + 1))

                h = self.W_i[self.voc.indexOf(sentence[i])]

                EH = np.array([0 for v in range(self.N)], dtype = np.float64)

                if self.hs:
                    for c in range(context_start, context_end):
                        if c != i:
                            curNode = self.ht.root
                            j = 0
                            while not curNode.word:
                                curNode.data -= (learning_rate * (Tools.sigmoid(np.dot(curNode.data, h)) - self.ht.t(sentence[c], j)) * h)
                                EH += ((Tools.sigmoid(np.dot(curNode.data, h)) - self.ht.t(sentence[c], j)) * curNode.data)
                                if self.ht.t(sentence[c], j): # left
                                    curNode = curNode.left
                                else:
                                    curNode = curNode.right
                                j += 1
                else:
                    u = np.matmul(h, self.W_o)
                    y = Tools.softmax(u)

                    # e = y - t
                    EI = np.array([0 for v in range(self.voc.size())], dtype = np.float64)
                    
                    context_cnt = 0
                    for c in range(context_start, context_end):
                        if c != i:
                            EI += (y - self.voc.vectorize(sentence[c]))

                    # update W_o
                    for j in range(self.voc.size()):
                        self.W_o[:, j] -= (learning_rate * EI[j] * h)

                    # update W_i
                    EH = np.matmul(EI, self.W_o.T)

                self.W_i[self.voc.indexOf(sentence[i])] -= (learning_rate * EH)    

    def train_cbow(self, learning_rate):
        for sentence in self.sentences:
            for i in range(len(sentence)):
                
                # context words
                context_start = max(0, int(i - self.C / 2))
                context_end = min(len(sentence), int(i + self.C / 2 + 1))
                context_sum = np.array([0 for v in range(self.N)], dtype = np.float64)
                context_cnt = 0
                
                for c in range(context_start, context_end):
                    if c != i:
                        context_sum += self.W_i[self.voc.indexOf(sentence[c])]
                        context_cnt += 1
                h = context_sum / context_cnt

                # self.E[sentence[i]] = math.log(np.exp(np.matmul(h, self.W_o)).sum(axis=0)) - np.dot(h, self.W_o[:, self.voc.indexOf(sentence[i])])
                
                EH = np.array([0 for v in range(self.N)], dtype = np.float64)

                # update Weights
                if self.hs:
                    curNode = self.ht.root
                    j = 0
                    while not curNode.word:
                        curNode.data -= (learning_rate * (Tools.sigmoid(np.dot(curNode.data, h)) - self.ht.t(sentence[i], j)) * h)
                        EH += ((Tools.sigmoid(np.dot(curNode.data, h)) - self.ht.t(sentence[i], j)) * curNode.data)
                        if self.ht.t(sentence[i], j): # left
                            curNode = curNode.left
                        else:
                            curNode = curNode.right
                        j += 1
                else:
                    # training word
                    t = self.voc.vectorize(sentence[i])
                    u = np.matmul(h, self.W_o)
                    y = Tools.softmax(u)
                    e = y - t

                    # update W_o
                    for j in range(self.voc.size()):
                        self.W_o[:, j] -= (learning_rate * e[j] * h)

                    # update W_i
                    EH = np.matmul(e, self.W_o.T)
                
                for c in range(context_start, context_end):
                    if c != i:
                        self.W_i[self.voc.indexOf(sentence[c])] -= (learning_rate * EH / context_cnt)

                
    def similarTo(self, word):
        if self.stem:
            stemmer = nltk.stem.porter.PorterStemmer()
            word = stemmer.stem(word)
        idx = self.voc.indexOf(word)
        l = [(self.voc.at(i), Tools.cosSimilarity(self.W_i[i], self.W_i[idx])) for i in range(self.voc.size()) if i != idx]
        l = sorted(l, key = lambda x: x[1])
        l.reverse()
        return l
'''
    def loss(self):
        E = 0   
        for sentence in self.sentences:
            for idx in range(len(sentence)):
                context_start = max(0, int(idx - self.C / 2))
                context_end = min(len(sentence), int(idx + self.C / 2 + 1))
                context_sum = np.array([0 for v in range(self.N)], dtype = np.float64)
                context_cnt = 0
                for c in range(context_start, context_end):
                    if c != idx:
                        context_sum += self.W_i[self.voc.indexOf(sentence[c])]
                        context_cnt += 1

                h = context_sum / context_cnt
                E += (math.log(np.exp(np.matmul(h, self.W_o)).sum(axis=0)) - np.dot(self.W_o[:, idx], h))


        E = E / self.voc.size()
        return E
'''