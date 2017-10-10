#!/usr/bin/env python

from copy import deepcopy
import math
import random
import os


class DecisionTree:
    def __init__(self, heuristic):
        self.attribute = None
        self.children = {}
        self.label = None
        self.header = None
        self.n_features = 0

        if heuristic == 'IG':
            self.heuristic = self.information_gain
        elif heuristic == 'VIG':
            self.heuristic = self.variance_impurity_gain

    def fit(self, X, y, header=None):
        self.header = header

        if len(X) == 0:
            self.label = 1
            return self

        self.n_features = len(X[0])
        if header is None:
            self.header = range(self.n_features)

        self.fit_recursively(X, y, [True for _ in range(self.n_features)])

        return self

    def predict(self, X):
        if self.is_leaf():
            return [self.label for _ in X]
        else:
            ret = [None for _ in X]
            for v, child in self.children.items():
                sub_X = [x for x in X if x[self.attribute] == v]
                preds = child.predict(sub_X)

                shift = 0
                for id_, x in enumerate(X):
                    if x[self.attribute] == v:
                        ret[id_] = preds[shift]
                        shift += 1

            return ret

    def fit_recursively(self, X, y, remaining_features):
        if len(set(y)) == 1:
            self.label = y[0]
        elif len(X) == 0 or sum(remaining_features) == 0:
            best_freq = -float('inf')
            self.label = None
            for v in range(2):
                next_freq = float(sum([yi == v for yi in y]))
                if next_freq > best_freq:
                    self.label = v
                    best_freq = next_freq
        else:
            self_copy = deepcopy(self)

            heuristic_per_ft = \
                [(self.heuristic(X, y, ft_id), ft_id)
                 for ft_id in range(self.n_features)
                 if remaining_features[ft_id]]
            heuristic_per_ft = sorted(heuristic_per_ft, reverse=True)

            best_feature = heuristic_per_ft[0][1]

            self.attribute = best_feature
            remaining_features[self.attribute] = False

            ft_values = [x[self.attribute] for x in X]
            for v in range(2):
                sub_X = [x for x, f in zip(X, ft_values) if f == v]
                sub_Y = [y_ for y_, f in zip(y, ft_values) if f == v]

                next_child = deepcopy(self_copy)
                next_child.fit_recursively(sub_X, sub_Y, remaining_features)
                self.children[v] = next_child

            remaining_features[self.attribute] = True

        return self

    def post_prunning(self, l, k, y, x):
        """
        :param l:
        :param k:
        :param y:
        :param x:
        :return:
        """
        d_best = deepcopy(self)
        max_accuracy = accuracy(y, d_best.predict(x))

        for i in range(l):
            d_temp = deepcopy(self)
            m = random.randrange(k) + 1

            for j in range(m):
                n = d_temp.number_of_non_leaves()
                p = random.randrange(n) + 1
                current = d_temp.get_nth_element(p, 0)
                if current is None:
                    continue
                label_temp, max_subset = current.max_subset()
                d_temp.set_nth_element(p, [0], label_temp)

            current_accuracy = accuracy(y, d_temp.predict(x))

            if current_accuracy >= max_accuracy:
                max_accuracy = current_accuracy
                d_best = deepcopy(d_temp)

        return d_best

    def get_label(self):
        return self.label

    def max_subset(self):
        val1 = 0
        val2 = 0
        for v, child in self.children.items():
            if child.is_leaf():
                if child.get_label() == 0:
                    val1 += 1
                else:
                    val2 += 1
            else:
                label_temp, val_temp = child.max_subset()
                if label_temp == 0:
                    val1 += val_temp
                else:
                    val2 += val_temp
        if val1 >= val2:
            return 0, val1
        return 1, val2

    def get_nth_element(self, n, current_id):
        if self.is_leaf():
            return None
        if current_id == n:
            return self
        for v, child in self.children.items():
            if child.is_leaf():
                continue
            current_id += 1
            res = self.get_nth_element(n, current_id)
            if res:
                return deepcopy(res)
            if current_id == n:
                return self
        return None

    def set_nth_element(self, n, current_id, label):
        if self.is_leaf():
            return False
        if current_id[0] == n:
            self.label = label
            return True
        for v, child in self.children.items():
            if child.is_leaf():
                continue
            current_id[0] += 1
            if child.set_nth_element(n, current_id, label):
                return True
            if current_id[0] == n:
                self.label = label
                return True
        return False

    def information_gain(self, X, y, ft_id):
        def entropy(y_):
            ret = [float(sum([yi_ == v for yi_ in y_])) /
                   float(len(y_)) for v in set(y_)]
            return float(sum([-pv * math.log(pv, 2) for pv in ret]))

        values = [x[ft_id] for x in X]
        gain = entropy(y)

        for v in set(values):
            vcount = sum([vi == v for vi in values])
            yv = [yv for yv, vv in zip(y, values) if vv == v]
            gain -= float(vcount) / float(len(X)) * entropy(yv)

        return gain

    def variance_impurity_gain(self, X, y, ft_id):
        def variance_impurity(y_):
            ret_list = [float(sum([yi_ == v for yi_ in y_])) /
                        float(len(y_)) for v in range(2)]
            ret = 1.
            for next_ret in ret_list:
                ret *= next_ret

            return ret

        values = [x[ft_id] for x in X]
        gain = variance_impurity(y)

        for v in set(values):
            vcount = float(sum([vi == v for vi in values])) / float(len(X))
            yv = [yv_ for yv_, vv in zip(y, values) if vv == v]
            gain -= vcount * variance_impurity(yv)

        return gain

    def print_tree(self, indent=0):
        def print_indent():
            return ' '.join(['|'] * indent) + ' 'if indent > 0 else ''

        if self.attribute is not None:
            for v, child in self.children.items():
                next_str = '%s%s = %d :' % (print_indent(),
                                            self.header[self.attribute], v)
                if child.is_leaf():
                    print(next_str)
                else:
                    print(next_str)

                child.print_tree(indent + 1)
        else:
            print(self.label)

    def is_leaf(self):
        return self.label is not None

    def number_of_leaves(self):
        if self.is_leaf():
            return 1
        leaves_num = 0
        if self.attribute is not None:
            for v, child in self.children.items():

                if child.is_leaf():
                    leaves_num += 1
                else:
                    leaves_num += child.number_of_leaves()

        return leaves_num

    def number_of_non_leaves(self):
        if self.is_leaf():
            return 0

        non_leaves_num = 1
        for v, child in self.children.items():
            if not child.is_leaf():
                non_leaves_num += child.number_of_non_leaves()

        return non_leaves_num

    def number_of_nodes(self):

        if self.is_leaf():
            return 1

        total = 1
        for v, child in self.children.items():
            total += child.number_of_nodes()

        return total


def accuracy(labels, predictions):
    if len(labels) == 0:
        return 1.

    correct = sum([float(l == p) for l, p in zip(labels, predictions)])

    return float(correct) / len(labels)


def read_dataset(filename):
    f = open(filename, 'r')
    lines = f.readlines()

    lines = [x.strip().split(',') for x in lines]

    header = lines[0][: -1]

    lines = [list(map(int, x)) for x in lines[1:]]

    X = [x[: -1] for x in lines]
    y = [x[-1] for x in lines]

    return header, X, y


L = int(os.sys.argv[1])
K = int(os.sys.argv[2])
tr_filename = os.sys.argv[3]
val_filename = os.sys.argv[4]
test_filename = os.sys.argv[5]
to_print = os.sys.argv[6]

tr_h, tr_X, tr_y = read_dataset(tr_filename)
val_h, val_X, val_y = read_dataset(val_filename)
ts_h, ts_X, ts_y = read_dataset(test_filename)

ig_dt = DecisionTree(heuristic='IG')
ig_dt.fit(tr_X, tr_y, header=tr_h)

vi_dt = DecisionTree(heuristic='VIG')
vi_dt.fit(tr_X, tr_y, header=tr_h)

ig_dt_post_pruned = ig_dt.post_prunning(L, K, val_y, val_X)
vi_dt_post_pruned = vi_dt.post_prunning(L, K, val_y, val_X)

if to_print.strip() == 'yes':
    print('Information Gain')
    ig_dt.print_tree()

    print()
    print('Variance Impurity')
    vi_dt.print_tree()


print()
print('Initial:')
print('Information Gain:  %.4f %.4f %.4f' % \
    (accuracy(tr_y, ig_dt.predict(tr_X)) * 100.,
     accuracy(val_y, ig_dt.predict(val_X)) * 100.,
     accuracy(ts_y, ig_dt.predict(ts_X)) * 100.))
print('Variance Impurity: %.4f %.4f %.4f' % \
    (accuracy(tr_y, vi_dt.predict(tr_X)) * 100.,
     accuracy(val_y, vi_dt.predict(val_X)) * 100.,
     accuracy(ts_y, vi_dt.predict(ts_X)) * 100.))
print()

print()
print('Post-pruned: L = %d, K = %d' % (L, K))
print('Information Gain:  %.4f %.4f %.4f' % \
    (accuracy(tr_y, ig_dt_post_pruned.predict(tr_X)) * 100.,
     accuracy(val_y, ig_dt_post_pruned.predict(val_X)) * 100.,
     accuracy(ts_y, ig_dt_post_pruned.predict(ts_X)) * 100.))
print('Variance Impurity: %.4f %.4f %.4f' % \
    (accuracy(tr_y, vi_dt_post_pruned.predict(tr_X)) * 100.,
     accuracy(val_y, vi_dt_post_pruned.predict(val_X)) * 100.,
     accuracy(ts_y, vi_dt_post_pruned.predict(ts_X)) * 100.))
print()


assert (ig_dt.number_of_non_leaves() + ig_dt.number_of_leaves() == ig_dt.number_of_nodes())
