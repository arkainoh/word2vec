import numpy as np
import math
import urllib.request
import nltk
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

"""
Created on Thu Apr  6 20:01:51 2017
@author: JAEMIN
"""

def sigmoid (x): return 1/(1 + np.exp(-x))

def findPath(number, v):
    n = int(math.log(v, 2))
    a = []
    if number <= 2*(v - 2**n):
        num = n
    else :
        number = number - (v - 2**n) 
        num = n-1
    while num >= 0:
        a.append(int(number/(2**num)))
        number = number%(2**num)
        num -= 1
    return a

    
v = 100  # voca size
n = 50 # hidden dimension
t_word = 72 #실제 out이 되어야하는 word의 번호

h = np.random.rand(n, 1)
w_o = np.random.rand(n, v)

a = findPath(t_word, v)
print(a)

n_th = 0
for i in a :
    if i == 0 :
        w_o[:, n_th] = w_o[:, n_th] - 0.2 * (sigmoid((w_o.transpose()[n_th].dot(h))[0]) - 1) * h[:, 0]
        print(n_th)
        n_th = n_th * 2 + 1
    else :
        w_o[:, n_th] = w_o[:, n_th] - 0.2 * (sigmoid((w_o.transpose()[n_th].dot(h))[0])) * h[:, 0]
        print(n_th)
        n_th = n_th * 2 + 2