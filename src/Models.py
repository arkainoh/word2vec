import numpy as np
import Tools
import nltk
import math

# implement CBOW, Skip-Gram
class cbow:

    def __init__(self, C, dimension, voc = None):
        self.W_i = np.empty(0)
        self.W_o = np.empty(0)
        self.C = C
        self.N = dimension
        self.voc = Tools.Vocabulary()
        if(voc is not None):
            self.voc.load(voc)
        self.sentences = []
        #self.E = {}

    def inputData(self, filename): # preprocessing
        inputstr = ""
        f = open(filename)
        lines = f.readlines()
        for line in lines:
            line = line.lower()
            for sentence in line.split('.'):
                tokens = nltk.word_tokenize(sentence)
                tokens = [i for i in tokens if i.isalpha()]
                if len(tokens) > 1:
                    self.sentences.append(tokens)
                    self.voc.add(tokens)

    def initWeights(self):
        V = self.voc.size()
        self.W_i = np.random.rand(V, self.N)
        self.W_o = np.random.rand(self.N, V)

    def train(self, learning_rate):
        for sentence in self.sentences:
            for i in range(len(sentence)):
                # training word
                t = self.voc.vectorize(sentence[i])
                
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

                u = np.matmul(h, self.W_o)
                y = Tools.softmax(u)
                # print(sentence[i], y, y.sum())
                e = y - t

                #self.E[sentence[i]] = ((e**2)/self.voc.size()).sum()
                # self.E[sentence[i]] = math.log(np.exp(np.matmul(h, self.W_o)).sum(axis=0)) - np.dot(h, self.W_o[:, self.voc.indexOf(sentence[i])])
                # update W_o
                for j in range(self.voc.size()):
                    self.W_o[:, j] -= (learning_rate * e[j] * h)

                # update W_i
                EH = np.matmul(e, self.W_o.T)
                
                for c in range(context_start, context_end):
                    if c != i:
                        self.W_i[self.voc.indexOf(sentence[c])] -= (learning_rate * EH / context_cnt)

                
    def similarTo(self, word):
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
class skipgram:

    def __init__(self, C, dimension, voc = None):
        self.W_i = np.empty(0)
        self.W_o = np.empty(0)
        self.C = C
        self.N = dimension
        self.voc = Tools.Vocabulary()
        if(voc is not None):
            self.voc.load(voc)
        self.sentences = []
        self.E = {}

    def inputData(self, filename):
        inputstr = ""
        f = open(filename)
        lines = f.readlines()
        for line in lines:
            line = line.lower()
            for sentence in line.split('.'):
                tokens = nltk.word_tokenize(sentence)
                tokens = [i for i in tokens if i.isalpha()]
                if len(tokens) > 1:
                    self.sentences.append(tokens)
                    self.voc.add(tokens)

    def initWeights(self):
        V = self.voc.size()
        self.W_i = np.random.rand(V, self.N)
        self.W_o = np.random.rand(self.N, V)

    def similarTo(self, word):
        idx = self.voc.indexOf(word)
        l = [(self.voc.at(i), Tools.cosSimilarity(self.W_i[i], self.W_i[idx])) for i in range(self.voc.size()) if i != idx]
        l = sorted(l, key = lambda x: x[1])
        l.reverse()
        return l

    def train(self, learning_rate):
        for sentence in self.sentences:
            for i in range(len(sentence)):
                # training word
                t = self.voc.vectorize(sentence[i])
                
                # context words
                h = self.W_i[self.voc.indexOf(sentence[i])]
                u = np.matmul(h, self.W_o)
                y = Tools.softmax(u)
                # e = y - t
                EI = np.array([0 for v in range(self.voc.size())], dtype = np.float64)
                context_start = max(0, int(i - self.C / 2))
                context_end = min(len(sentence), int(i + self.C / 2 + 1))
                context_cnt = 0
                for c in range(context_start, context_end):
                    if c != i:
                        EI += (y - self.voc.vectorize(sentence[c]))

                # self.E[sentence[i]] = math.log(np.exp(np.matmul(h, self.W_o)).sum(axis=0)) - np.dot(h, self.W_o[:, self.voc.indexOf(sentence[i])])

                # update W_o
                for j in range(self.voc.size()):
                    self.W_o[:, j] -= (learning_rate * EI[j] * h)

                # update W_i
                EH = np.matmul(EI, self.W_o.T)
                
                self.W_i[self.voc.indexOf(sentence[i])] -= (learning_rate * EH)

'''
    def loss(self, idx):
        context_start = max(0, int(idx - self.C / 2))
        context_end = min(len(self.words), int(idx + self.C / 2 + 1))
        context_sum = np.array([0 for v in range(self.N)], dtype = np.float64)
        context_cnt = 0
        for c in range(context_start, context_end):
            if c != idx:
                context_sum += self.W_i[self.voc.indexOf(self.words[c])]
                context_cnt += 1

        h = context_sum / context_cnt
        E = math.log(np.exp(np.matmul(h, self.W_o)).sum(axis=0)) - np.dot(self.W_o[:, idx], h)
        return E
'''
