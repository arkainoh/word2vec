import numpy as np
import Tools
import nltk

# implement CBOW, Skip-Gram
class CBOW:

	def __init__(self, C, N):
		self.W_i = np.empty(0)
		self.W_i = np.empty(0)
		self.C = C
		self.N = N
		self.V = 0
		self.voc = Tools.Vocabulary()

	def setTrainingData(self, filename):
		inputstr = ""
		f = open(filename)
		lines = f.readlines()
		for line in lines:
			inputstr += line
		inputstr = inputstr.lower()
		tokens = nltk.word_tokenize(inputstr)
		tokens = [i for i in tokens if i.isalpha()]
		self.voc.add(tokens)
		self.V = self.voc.size()
		self.W_i = np.random.rand(self.V, self.N)
		self.W_o = np.random.rand(self.N, self.V)
	
	def train(self):
		for i in range(self.V):
			h = np.transpose(self.W_i[i]) # hidden layer
			u = np.matmul(h, self.W_o)
			y = Tools.softmax(u)
			for j in range(self.V):
				# update W_o_j
				t = voc.vectorize(j)
			# update W_i

test = CBOW(C = 5, N = 5)
test.setTrainingData('test.txt')
test.train()
