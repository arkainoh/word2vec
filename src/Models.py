import numpy as np
import Tools
import nltk
import math

# implement CBOW, Skip-Gram
class CBOW:

    def __init__(self, C, dimension, voc = None):
        self.W_i = np.empty(0)
        self.W_o = np.empty(0)
        self.C = C
        self.N = dimension
        self.voc = Tools.Vocabulary()
        if(voc is not None):
            self.voc.load(voc)
        self.V = self.voc.size()
        self.words = []

    def addTrainingData(self, filename):
        inputstr = ""
        f = open(filename)
        lines = f.readlines()
        for line in lines:
            inputstr += line
        inputstr = inputstr.lower()
        tokens = nltk.word_tokenize(inputstr)
        tokens = [i for i in tokens if i.isalpha()]
        self.voc.add(tokens)
        self.words += tokens
        self.V = self.voc.size()

    def init_weights(self):
        self.W_i = np.random.rand(self.V, self.N)
        self.W_o = np.random.rand(self.N, self.V)

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


    def train(self, learning_rate):
        for i in range(len(self.words)):

            # training word
            t = self.voc.vectorize(self.words[i])
            
            # context words
            context_start = max(0, int(i - self.C / 2))
            context_end = min(len(self.words), int(i + self.C / 2 + 1))
            context_sum = np.array([0 for v in range(self.N)], dtype = np.float64)
            context_cnt = 0
            for c in range(context_start, context_end):
                if c != i:
                    context_sum += self.W_i[self.voc.indexOf(self.words[c])]
                    context_cnt += 1

            h = context_sum / context_cnt

            u = np.matmul(h, self.W_o)
            y = Tools.softmax(u)
            e = y - t

            # update W_o
            for j in range(self.V):
                self.W_o[:, j] -= (learning_rate * e[j] * h)

            # update W_i
            EH = np.matmul(e, self.W_o.T)
            
            for c in range(context_start, context_end):
                if c != i:
                    self.W_i[self.voc.indexOf(self.words[c])] -= (learning_rate * EH / context_cnt)
