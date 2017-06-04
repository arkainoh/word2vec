import numpy as np
import Tools
import nltk
from nltk.corpus import stopwords
import math
import re

# implement CBOW, Skip-Gram

class word2vec:

    def __init__(self, model, dimension, C, filename = '',
        hs = True, min_count = 0,
        lowercase = True, special_letters = False, tokenizer = True, stopwords = False, stemmer = True):
    
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
        self.lowercase = lowercase
        self.special_letters = special_letters
        self.tokenizer = tokenizer
        self.stopwords = stopwords
        self.stemmer = stemmer
        if model is 'cbow':
            self.model = 0
        elif model is 'skipgram':
            self.model = 1
        else:
            print("<ERROR> model should be either \'cbow\' or \'skipgram\'")
            raise ValueError

        if(filename):
            self.inputData(filename)

    def inputData(self, filename): # preprocessing
        inputstr = ""
        words = []
        stpwrds = None
        stemmer = None
        if not self.stopwords:
            stpwrds = set(stopwords.words('english'))
        if self.stemmer:
            stemmer = nltk.stem.porter.PorterStemmer()
        f = open(filename)
        lines = f.readlines()
        for line in lines:
            if self.lowercase:
                line = line.lower()
            for sentence in re.split('\.|\!|\?|;|:', line):
                tmp = []
                if not self.special_letters:
                    sentence = sentence.replace("'", " ")
                    sentence = sentence.replace("\"", " ")
                if self.tokenizer:
                    tmp = nltk.word_tokenize(sentence)
                else:
                    tmp = sentence.split(' ')
                tokens = []
                for token in tmp:
                    if not self.special_letters:
                        if not token.isalpha():
                            continue
                    if not self.stopwords:
                        if token in stpwrds:
                            continue
                    if self.stemmer:
                        token = stemmer.stem(token)
                    if len(token):
                        tokens.append(token)

                if len(tokens) > 1:
                    self.sentences.append(tokens)
                    for token in tokens:
                        words.append(token)
                    if self.min_count == 0:
                        self.voc.add(tokens)

        self.freq = Tools.tokFreq(words)

        if self.min_count > 0:
            tmp = self.sentences
            self.sentences = []
            words = []
            for sentence in tmp:
                sentence = [w for w in sentence if self.freq[w] >= self.min_count]
                if len(sentence) > 1:
                    self.voc.add(sentence)
                    self.sentences.append(sentence)
                    for word in sentence:
                        words.append(word)
            self.freq = Tools.tokFreq(words)

        if self.hs:
            self.ht = Tools.HuffmanTree(self.N, self.freq)
        f.close()
        self.initWeights()

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
        if self.stemmer:
            stemmer = nltk.stem.porter.PorterStemmer()
            word = stemmer.stem(word)
        idx = self.voc.indexOf(word)
        l = [(self.voc.at(i), Tools.cosSimilarity(self.W_i[i], self.W_i[idx])) for i in range(self.voc.size()) if i != idx]
        l = sorted(l, key = lambda x: x[1])
        l.reverse()
        return l

    def vectorOf(self, word):
        return self.W_i[self.voc.indexOf(word)]

    def probabilityOf(self, word, contextwords = []):
        if not len(contextwords):
            print("<ERROR> The number of context words should not be 0")
            raise ValueError
            return None
        idx = self.voc.indexOf(word)
        context_sum = np.array([0 for v in range(self.N)], dtype = np.float64)
        context_cnt = 0

        for word in contextwords:
            context_sum += self.W_i[self.voc.indexOf(word)]
            context_cnt += 1

        h = context_sum / context_cnt
        if self.hs:
            p_w = 1
            curNode = self.ht.root
            j = 0
            while not curNode.word:
                if self.ht.t(self.voc.at(idx), j):
                    p_w *= Tools.sigmoid(np.dot(curNode.data, h))
                    curNode = curNode.left
                else:
                    p_w *= Tools.sigmoid(-np.dot(curNode.data, h))
                    curNode = curNode.right
                j += 1
        else:
            u = np.matmul(h, self.W_o)
            y = Tools.softmax(u)
            p_w = y[idx]
        return p_w

    def recommend(self, contextwords = []):
        if not len(contextwords):
            print("<ERROR> The number of context words should not be 0")
            raise ValueError
            return None

        result = []
        context_sum = np.array([0 for v in range(self.N)], dtype = np.float64)
        context_cnt = 0

        for word in contextwords:
            context_sum += self.W_i[self.voc.indexOf(word)]
            context_cnt += 1

        h = context_sum / context_cnt

        if self.hs:
            for i in range(self.voc.size()):
                p_w = 1
                curNode = self.ht.root
                j = 0
                while not curNode.word:
                    if self.ht.t(self.voc.at(i), j):
                        p_w *= Tools.sigmoid(np.dot(curNode.data, h))
                        curNode = curNode.left
                    else:
                        p_w *= Tools.sigmoid(-np.dot(curNode.data, h))
                        curNode = curNode.right
                    j += 1
                result.append((self.voc.at(i), p_w))
        else:
            u = np.matmul(h, self.W_o)
            y = Tools.softmax(u)
            result = [(self.voc.at(i), y[i]) for i in range(self.voc.size())]

        result = sorted(result, key = lambda x: x[1])
        result.reverse()
        return result

    def multiRecommend(self, num, left_contextwords, right_contextwords):
        if (not len(left_contextwords)) and (not len(right_contextwords)):
            print("<ERROR> The number of context words should not be 0")
            raise ValueError
            return None

        if num < 1 or num > 4:
            print("<ERROR> The number of code should range from 1 to 4")
            raise ValueError
            return None

        result = []

        if num == 1:
            result = self.recommend(left_contextwords + right_contextwords)[:10]
            result = [item for item in result if item[1] > 0.1]
            for item in result:
                similar_word = self.similarTo(item[0])[0][0]
                contain = False
                for element in result:
                    if similar_word in element:
                        contain = True
                        break
                if not contain:
                    result.append((similar_word, self.probabilityOf(similar_word, left_contextwords + right_contextwords)))
                
            result = sorted(result, key = lambda x: x[1])
            result.reverse()
            return result

        leftmost = []
        rightmost = []
        if num == 2:
            leftmost = self.recommend(left_contextwords + [right_contextwords[0]])[:10]
            rightmost = self.recommend([left_contextwords[1]] + right_contextwords)[:10]
        else:
            leftmost = self.recommend(left_contextwords)[:10]
            rightmost = self.recommend(right_contextwords)[:10]

        leftmost = [item for item in leftmost if item[1] > 0.1]
        rightmost = [item for item in rightmost if item[1] > 0.1]

        for item in leftmost:
                similar_word = self.similarTo(item[0])[0][0]
                contain = False
                for element in leftmost:
                    if similar_word in element:
                        contain = True
                        break
                if not contain:
                    if num == 2:
                        leftmost.append((similar_word, self.probabilityOf(similar_word, left_contextwords + [right_contextwords[0]])))
                    else:
                        leftmost.append((similar_word, self.probabilityOf(similar_word, left_contextwords)))

        for item in rightmost:
                similar_element = self.similarTo(item[0])[0][0]
                contain = False
                for element in rightmost:
                    if similar_word in element:
                        contain = True
                        break
                if not contain:
                    if num == 2:
                        rightmost.append((similar_word, self.probabilityOf(similar_word, [left_contextwords[1]] + right_contextwords)))
                    else:
                        rightmost.append((similar_word, self.probabilityOf(similar_word, right_contextwords)))

        if num == 2:
            for i in leftmost:
                for j in rightmost:
                    codes = []
                    codes.append(i[0])
                    codes.append(j[0])
                    result.append((codes, i[1] * j[1]))
            result = sorted(result, key = lambda x: x[1])
            result.reverse()
            return result

        remainingNum = num - 2

        for i in leftmost:
            for j in rightmost:
                left_codes = []
                right_codes = []
                left_codes.append(left_contextwords[1])
                left_codes.append(i[0])
                right_codes.append(right_contextwords[0])
                right_codes.append(j[0])
                
                center = self.multiRecommend(remainingNum, left_codes, right_codes)
                for item in center:
                    codes = []
                    codes.append(i[0])
                    if remainingNum == 2:
                        codes = codes + item[0]
                    else:
                        codes.append(item[0])
                    codes.append(j[0])
                    result.append((codes, i[1] * j[1] * item[1]))

        result = sorted(result, key = lambda x: x[1])
        result.reverse()
        return result[:10]

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
