import numpy as np
import Tools
import nltk

# implement CBOW, Skip-Gram
class CBOW:

    def __init__(self, C, dimension, voc):
        self.W_i = np.empty(0)
        self.W_o = np.empty(0)
        self.C = C
        self.N = dimension
        self.voc = Tools.Vocabulary()
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
        self.words.append(tokens)
        self.V = self.voc.size()

    def init_weights():
        self.W_i = np.random.rand(self.V, self.N)
        self.W_o = np.random.rand(self.N, self.V)
    
    def train(self, learning_rate):
        for i in range(self.V):

            # collect C words
            # self.W_i[i] = avg(C words)

            t = self.voc.vectorize(self.voc.at(i)) # target word

            # h = self.W_i[i]
            u = np.matmul(self.W_i[i], self.W_o)
            y = Tools.softmax(u)
            
            # update W_o
            for j in range(self.V):
                e = y - t
                self.W_o[:, j] -= learning_rate * e[j] * self.W_i[i]
            
            # update W_i
            EH = np.matmul(e, self.W_o.T)
            self.W_i[i] -= learning_rate * EH

