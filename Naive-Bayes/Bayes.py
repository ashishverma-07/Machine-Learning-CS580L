#!/usr/bin/env python

import os
import string


class NaiveBayes(object):
    """
        multinomial Naive Bayes algorithm for text classification
        http://nlp.stanford.edu/IR-book/pdf/13bayes.pdf
    """
    def __init__(self):
        self.ham_dict = {}
        self.spam_dict = {}
        self.num_of_ham = 0
        self.num_of_spam = 0

    def extract_vocabulary(self, ham_path, spam_path, stopwords):

        for filename in os.listdir(ham_path):
            self.num_of_ham += 1
            words = read_dataset(ham_path + filename)
            for w in words:
                if w in stopwords:
                    continue
                if self.ham_dict.get(w, 0) == 0:
                    self.ham_dict[w] = 1
                else:
                    self.ham_dict[w] += 1

        for filename in os.listdir(spam_path):
            self.num_of_spam += 1
            words = read_dataset(spam_path + filename)
            for w in words:
                if w in stopwords:
                    continue
                if self.spam_dict.get(w, 0) == 0:
                    self.spam_dict[w] = 1
                else:
                    self.spam_dict[w] += 1

    def predict(self, test_path):
        labels = []
        for filename in os.listdir(test_path):
            words = read_dataset(test_path + filename)
            current_ham_prob = 1
            current_spam_prob = 1
            for w in words:
                if self.ham_dict.get(w, 0) + self.spam_dict.get(w, 0) < 4:
                    continue
                prob = self.ham_dict.get(w, 0) / (self.ham_dict.get(w, 0) + self.spam_dict.get(w, 0))
                current_ham_prob *= prob
                current_spam_prob *= 1 - prob
            if current_ham_prob >= current_spam_prob:
                labels.append(0)
            else:
                labels.append(1)
        assert labels
        return labels


def accuracy(spam, ham):
    total = len(spam) + len(ham)
    right = sum(1 for x in spam if x == 1) + sum(1 for x in ham if x == 0)
    assert total
    assert right
    result = float(right) / float(total)
    return result


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

bayes = NaiveBayes()
stop_words = load_stopwords("stopwords.txt")
assert stop_words
bayes.extract_vocabulary(train_ham_path, train_spam_path, stop_words)
assert bayes.ham_dict
assert bayes.spam_dict
ham_labels = bayes.predict(test_ham_path)
assert ham_labels
spam_labels = bayes.predict(test_spam_path)
assert spam_labels
final_accuracy = 100*accuracy(spam_labels, ham_labels)
assert final_accuracy
print('Total accuracy: %.4f' % final_accuracy)
