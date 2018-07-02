from collections import defaultdict
from itertools import chain
from math import log

class NaiveBayes:

    def __init__(self, featureset, featureset_class, binarization=False):
        """
        The constructor accepts two basic parameters for
        the training set and its corresponding class.

        e.g.

        >>> A = NaiveBayes(["cogito, ergo sum", "νόησις νοήσεως"], ["Latin", "Greek"])

        :param text_class: str list
        :param text_labels: str list
        """

        self.N_features = len(featureset)

        if binarization:
            print(featureset)
            for i in range(self.N_features):
                featureset[i] = list(set(featureset[i]))
            print(featureset)

        featureset_class = tuple(featureset_class)

        labeled = defaultdict(list)

        for feature, c in zip(featureset, featureset_class):
            try:
                labeled[c].append(feature)
            except ValueError:
                labeled[c] = []
                labeled[c].append(feature)

        self.prior_probability = {c: self.prior(c, labeled) for c in featureset_class}

        words = list(chain(*featureset))
        self.N = len(set(words))

        self.likelihood_probability = defaultdict(dict)

        for c in set(featureset_class):
            class_words = list(chain(*labeled[c]))
            for w in words:
                self.likelihood_probability[c][w] = self.likelihood( w, class_words)

    def get_likelihood(self, c, w):
        return self.likelihood_probability[c].setdefault(w,
                                                      log(1/(len(self.likelihood_probability[c]) + self.N)))

    def prior(self, c, labeled):
        """
        Calculate the prior probability P(c),
        P(c) = occurrences of class c / total documents

        :param c: str
        :param labeled: dictionary: dictionary of the form
        {class: features}
        """
        return log(len(labeled[c])/self.N_features)

    def likelihood(self, w, class_words):
        """
        Calculates the probability of a word w belonging to
        the class c.

        :param c: str
        :param w: str
        :param class_words: total number of words belonging to the class c
        """
        return log((class_words.count(w) + 1)/(len(class_words) + self.N))

    def probability_s(self, s, c):
        """
        Probability a sentence s belongs to class c
        :param s: str
        :param c: str
        """
        return sum([self.get_likelihood(c, w) for w in s]) + self.prior_probability[c]

    def predict(self, s):
        """
        Predict class of sentence s
        :param s: str
        """
        return max([(self.probability_s(s, c), c) for c in self.likelihood_probability.keys()])[1]

