import numpy as np
from Tools import Vocabulary
import nltk

# implement CBOW, Skip-Gram
class CBOW:

	def __init__(self, N):
		self.W_i = np.empty(0)
		self.W_i = np.empty(0)
		self.N = N
		self.V = 0
		self.voc = Vocabulary()

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



