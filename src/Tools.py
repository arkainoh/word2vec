import numpy as np
import math
import heapq

# implement Softmax, Negative Sampling and other useful functions

def softmax(vec):
    m = np.max(vec)
    return np.exp(vec - m) / np.exp(vec - m).sum(axis = 0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
    
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
    def __init__(self, word = '', data = None, left = None, right = None, weight = 0):
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
    def __init__(self, dimension, tokfreq = None):
        self.dimension = dimension
        self.root = None
        self.path = {}
        if(tokfreq):
            self.build(tokfreq)

    def build(self, tokfreq): # build Huffman tree with given term frequencies
        pq = []
        for key in tokfreq:
            pq.append(TreeNode(word=key, weight=tokfreq[key]))
            self.path[key] = []

        heapq.heapify(pq)
        while(len(pq) >= 2):
            tmpR = heapq.heappop(pq)
            tmpL = heapq.heappop(pq)
            tmpP = TreeNode(data = np.random.rand(self.dimension), left = tmpL, right = tmpR, weight = tmpR.weight + tmpL.weight)

            # this algorithm seems inefficient... it should be replaced later
            for word in self.words(tmpR):
                self.path[word].append(0)
            for word in self.words(tmpL):
                self.path[word].append(1)

            heapq.heappush(pq, tmpP)
        self.root = pq[0]
        for key in self.path:
            self.path[key].reverse()

    def nodes(self, node = None): # inorder traversal
        l = []
        stack = []
        if(node):
            curNode = node
        else:
            curNode = self.root
        while(len(stack) or curNode is not None):
            if curNode is not None:
                stack.append(curNode)
                curNode = curNode.left
            else:
                curNode = stack.pop()
                l.append(curNode)
                curNode = curNode.right
        return l

    def words(self, node = None): # inorder traversal
        l = []
        stack = []
        if(node):
            curNode = node
        else:
            curNode = self.root
        while(len(stack) or curNode is not None):
            if curNode is not None:
                stack.append(curNode)
                curNode = curNode.left
            else:
                curNode = stack.pop()
                if curNode.word:
                    l.append(curNode.word)
                curNode = curNode.right
        return l

    # j starts with 0 in n(w, j) and t(w, j)
    def n(self, word, j):
        if j > len(self.path[word]):
            return None
        curNode = self.root
        for i in range(j):
            if self.path[word][i]: # left
                curNode = curNode.left
            else:
                curNode = curNode.right
        return curNode

    def t(self, word, j):
        return self.path[word][j]
