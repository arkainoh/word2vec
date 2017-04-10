import numpy as np
# implement Softmax, Negative Sampling and other useful functions

def softmax(vec):
	m = np.max(vec)
	return np.exp(vec - m) / np.exp(vec - m).sum(axis = 0)

# Set of vocabularies with indices
class Vocabulary:
	def __init__(self):
		self.vector = {}

	def add(self, tokens):
		for token in tokens:
			if token not in self.vector:
				self.vector[token] = len(self.vector)

	def indexOf(self, vocab):
		return self.vector[vocab]

	def size(self):
		return len(self.vector)

	def at(self, i):
		return self.vector[] # fix this!

	# vectorize = dict -> numpy.array
	def vectorize(self, word):
		v = [0 for i in range(self.size())]
		if word in self.vector:
			v[self.indexOf(word)] = 1
		else:
			print("<ERROR> Word \'" + word + "\' Not Found")
		return np.array(v)

	def __str__(self):
		s = "Vocabulary("
		for word in self.vector:
			s += (str(self.vector[word]) + ": " + word + ", ")
		if self.size() != 0:
			s = s[:-2]
		s += ")"
		return s
