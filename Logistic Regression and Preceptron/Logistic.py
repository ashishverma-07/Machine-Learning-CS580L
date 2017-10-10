#!/usr/bin/env python

import os
import math
import string


class LogisticRegression(object):
    """
        multinomial LogisticRegression algorithm for text classification
        http://www.cs.cmu.edu/%7Etom/mlbook/NBayesLogReg.pdf
    """
    def __init__(self, eta, Lambda, niter=100):
        
        self.niter = niter
        self.eta = eta
        self.Lambda = Lambda
        self.w = []

    def train(self, X_train, Y_train, nfeatures):
        
        n = len(X_train)
        m = nfeatures
        X_train_tr = transpose(X_train, m)
        
        self.w = [0] * m
        for k in range(self.niter):
            pred_error = [ sigmoid( dot_product(X_train[i], self.w) ) - Y_train[i] for i in range(n) ]
            gradient = [ self.eta * dot_product(X_train_tr[j], pred_error) for j in range(m) ]
            L2 =  [ self.eta * self.Lambda * self.w[j] for j in range(m) ]
            self.w = [ self.w[j] - gradient[j] + L2[j] for j in range(m) ]

    def predict(self, X_test):
        probability = [ sigmoid( dot_product(X_test[i], self.w) ) for i in range(len(X_test)) ]
        return [int(p > 0.5) for p in probability]



class Perceptron(object):

    def __init__(self, eta, niter=100):
        
        self.niter = niter
        self.eta = eta
        self.w = []
    
    def train(self, X_train, Y_train, nfeatures):
        
        n = len(X_train)
        m = nfeatures
        X_train_tr = transpose(X_train, m)
        
        self.w = [0] * m
        for k in range(self.niter):
            for i in range(n):
                o = int(dot_product(X_train[i], self.w) > 0)
                for j in X_train[i]:
                    self.w[j] += self.eta * (Y_train[i] - o) * X_train[i][j]
            
    def predict(self, X_test):
        return [ int(dot_product(X_test[i], self.w) > 0) for i in range(len(X_test)) ]
        


def get_column(arr, index):
    return [row[index] for row in arr]


def dot_product(sparse, arr):
    # dot product of the normal array and sparse array
    return sum(arr[k] * sparse[k] for k in sparse)


def transpose(arr, ncols):
    # arr is a list of dicts -- sparse matrix
    # ncols is a number of columns in arr
    transposed = []
    for j in range(ncols):
        row = {i: row[j] for i, row in enumerate(arr) if j in row}
        transposed.append(row)
    return transposed                
    

def sigmoid(x):
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0


def accuracy(predicted, actual):
    total = len(predicted)
    right = sum( int(predicted[i] == actual[i]) for i in range(total))
    assert total
    assert right
    result = float(right) / float(total)
    return result
    
    
def extract_vocabulary(the_path, all_words, stopwords):

    features = []
    for filename in os.listdir(the_path):
        words = read_dataset(the_path + filename)
        the_dict = {}
        for w in words:
            if w in stopwords:
                continue
            if the_dict.get(w, 0) == 0:
                the_dict[w] = 1
            else:
                the_dict[w] += 1
                
        #row = [the_dict.get(w, 0) for w in all_words]
        row = {all_words.index(w) + 1: the_dict[w] for w in the_dict}
        row[0] = 1   # add a column of ones for w_0
        features.append(row)
    return features   


def extract_words(target_dir, stopwords):
    
    all_words = set()
    for path in target_dir:
        for filename in os.listdir(path):
            words = read_dataset(path + filename)
            for w in words:
                if not w in stopwords:
                    all_words.add(w)
                
    all_words = sorted(list(all_words))     
    return all_words               
        

def read_dataset(file_path):
    words = []
    f = open(file_path, 'r')

    words = [word.strip(string.punctuation).lower() for line in f for word in line.split()
             if sum(1 for c in word if c.islower()) > 1]

    assert words
    return words


def load_stopwords(file_path):  
    words = []
    f = open(file_path, 'r')

    words = [word for line in f for word in line.split()]

    return words



train_ham_path = os.sys.argv[1]
train_spam_path = os.sys.argv[2]
test_ham_path = os.sys.argv[3]
test_spam_path = os.sys.argv[4]

stop_words = load_stopwords("stopwords.txt")
assert stop_words

#stop_words = []

target_dir = [train_ham_path, train_spam_path, test_ham_path, test_spam_path]
words      = extract_words(target_dir, stop_words)
assert words

train_ham  = extract_vocabulary(train_ham_path, words, stop_words)
train_spam = extract_vocabulary(train_spam_path, words, stop_words)
test_ham   = extract_vocabulary(test_ham_path, words, stop_words)
test_spam  = extract_vocabulary(test_spam_path, words, stop_words)

X_train = train_ham + train_spam
assert X_train
X_test  = test_ham + test_spam
assert X_test
Y_train = [0] * len(train_ham) + [1] * len(train_spam)
assert Y_train
Y_test  = [0] * len(test_ham)  + [1] * len(test_spam)
assert Y_test


print('Logistic Regression:')
for Lambda in [0.1, 1, 10, 100, 1000]:
    logistic = LogisticRegression(0.005, Lambda, 50)
    logistic.train(X_train, Y_train, len(words) + 1)
    pred = logistic.predict(X_test)
    print("Lambda = {}  accuracy = {:.4f}".format(Lambda, accuracy(pred, Y_test)) )
    


print('Perceptron:')
for niter in [5, 10, 20, 50]:
    for eta in [0.0001, 0.01, 0.05, 0.1, 0.5]:
        perceptron = Perceptron(eta, niter)
        perceptron.train(X_train, Y_train, len(words) + 1)
        pred = perceptron.predict(X_test)
        print("n_iter = {}   eta = {}   accuracy = {:.5f}".format(niter, eta, accuracy(pred, Y_test)) )    
