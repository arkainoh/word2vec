import numpy as np
import math
import heapq
# implement Softmax, Negative Sampling and other useful functions

def softmax(vec):
    m = np.max(vec)
    return np.exp(vec - m) / np.exp(vec - m).sum(axis = 0)
    
def cosSimilarity(A, B):
    multi = (A.dot(B))
    x = math.sqrt(A.dot(A))
    y = math.sqrt(B.dot(B))
    result = multi / (x * y)
    return result

def tokFreq(tokens):
    dic = {}
    for token in tokens:
        if token in dic:
            dic[token] += 1
        else:
            dic[token] = 1
    return dic

# Set of vocabularies with indices
class Vocabulary:
    def __init__(self):
        self.vector = {}

    def add(self, tokens):
        for token in tokens:
            if token not in self.vector and not token.isspace() and token != '':
                self.vector[token] = len(self.vector)

    def indexOf(self, vocab):
        return self.vector[vocab]

    def size(self):
        return len(self.vector)

    def at(self, i): # get ith word in the vector
        return list(self.vector)[i]

    # vectorize = dict -> numpy.array
    def vectorize(self, word):
        v = [0 for i in range(self.size())]
        if word in self.vector:
            v[self.indexOf(word)] = 1
        else:
            print("<ERROR> Word \'" + word + "\' Not Found")
        return np.array(v)

    def save(self, filename):
        f = open(filename, 'w', encoding='utf-8')
        for word in self.vector:
            f.write(word + '\n')

    def load(self, filename):
        f = open(filename, 'r', encoding='utf-8')
        lines = f.readlines()
        bow = [i[:-1] for i in lines]
        self.add(bow)

    def __str__(self):
        s = "Vocabulary("
        for word in self.vector:
            s += (str(self.vector[word]) + ": " + word + ", ")
        if self.size() != 0:
            s = s[:-2]
        s += ")"
        return s

class TreeNode:
    def __init__(self, word='', data=None, left=None, right=None, weight=0):
        self.word = word
        self.data = data
        self.left = left
        self.right = right
        self.weight = weight

    def __lt__(self, other):
        return self.weight < other.weight

    def __le__(self, other):
        return self.weight <= other.weight

class HuffmanTree:
    def __init__(self, dimension, root=None):
        self.dimension = dimension
        self.root = None
    def build(self, tokfreq): # build Huffman tree with given term frequencies
        pq = [TreeNode(word=key, weight=tokfreq[key]) for key in tokfreq]
        heapq.heapify(pq)
        while(len(pq) >= 2):
            tmpR = heapq.heappop(pq)
            tmpL = heapq.heappop(pq)
            heapq.heappush(pq, TreeNode(data=np.random.rand(self.dimension), left=tmpL, right=tmpR, weight=tmpR.weight + tmpL.weight))
        self.root = pq[0]

def sigmoid (x): return 1 / (1 + np.exp(-x))

def findPath(number, v):
    n = int(math.log(v, 2))
    a = []
    if number <= 2 * (v - 2 ** n):
        num = n
    else :
        number = number - (v - 2 ** n) 
        num = n - 1
    while num >= 0:
        a.append(int(number / (2 ** num)))
        number = number % (2 ** num)
        num -= 1
    return a

def jaemin():

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

