from collections import OrderedDict, Counter
from sklearn.base import TransformerMixin
from typing import List, Union
from nltk import RegexpTokenizer
import numpy as np


class BoW(TransformerMixin):
    """
    Bag of words tranformer class
    
    check out:
    https://scikit-learn.org/stable/modules/generated/sklearn.base.TransformerMixin.html
    to know about TransformerMixin class
    """

    def __init__(self, k: int):
        """
        :param k: number of most frequent tokens to use
        """
        self.k = k
        # dict of (k most frequent tokens, their id)
        self.bow = None
        self.tokenizer = RegexpTokenizer(r"\w+")

    def fit(self, X: np.ndarray, y=None):
        """
        :param X: array of texts to be trained on
        """
        # task: find up to self.k most frequent tokens in texts_train,
        # sort them by number of occurences (highest first)
        # store most frequent tokens in self.bow
        c = Counter(self.tokenizer.tokenize(' '.join(X)))
        if self.k is None:
            c = sorted(c.keys(), key=lambda x : c[x], reverse=True)
        else:
            c = sorted(c.keys(), key=lambda x : c[x], reverse=True)[:self.k]
        self.bow = {token : i for i, token in enumerate(c)}
        # fit method must always return self
        return self

    def _text_to_bow(self, text: str) -> np.ndarray:
        """
        convert text string to an array of token counts. Use self.bow.
        :param text: text to be transformed
        :return bow_feature: feature vector, made by bag of words
        """

        result = np.zeros(len(self.bow))
        text = text.split()
        for token in text:
            if token in self.bow.keys():
                result[self.bow[token]] += 1
        
        return result

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        :param X: array of texts to transform
        :return: array of transformed texts
        """
        assert self.bow is not None
        return np.stack([self._text_to_bow(text) for text in X])

    def get_vocabulary(self) -> Union[List[str], None]:
        return list(self.bow.keys())


class TfIdf(TransformerMixin):
    """
    Tf-Idf tranformer class
    if you have troubles implementing Tf-Idf, check out:
    https://streamsql.io/blog/tf-idf-from-scratch
    """
    def __init__(self, k: int = None, normalize: bool = False, alpha: int = 1):
        """
        :param k: number of most frequent tokens to use
        if set k equals None, than all words in train must be considered
        :param normalize: if True, you must normalize each data sample
        after computing tf-idf features
        """
        self.k = k
        self.normalize = normalize
        self.alpha = alpha

        # self.idf[term] = log(total # of documents / # of documents with term in it)
        self.idf = None
        self.bow = None
        self.tokenizer = RegexpTokenizer(r"\w+")

    def fit(self, X: np.ndarray, y=None):
        """
        :param X: array of texts to be trained on
        """
        c = Counter(self.tokenizer.tokenize(' '.join(X)))
        if self.k is None:
            c = sorted(c.keys(), key=lambda x : c[x], reverse=True)
        else:
            c = sorted(c.keys(), key=lambda x : c[x], reverse=True)[:self.k]
            
        def calc_idf(token, tokenizer):
            count = 0
            for s in X:
                if token in s.split():
                    count += 1
            return np.log(X.shape[0] / (count + self.alpha))
            
        self.bow = {token : i for i, token in enumerate(c)}
        self.idf = {token : calc_idf(token, self.tokenizer) for token in c}
        
        # fit method must always return self
        return self

    def _text_to_tf_idf(self, text: str) -> np.ndarray:
        """
        convert text string to an array tf-idfs.
        *Note* don't forget to normalize, when self.normalize == True
        :param text: text to be transformed
        :return tf_idf: tf-idf features
        """
        result = np.zeros(len(self.bow))
        text = text.split()        
        for token in text:
            if token in self.bow.keys():
                tf = text.count(token) / len(text)
                result[self.bow[token]] +=  tf * self.idf[token]
        
        if self.normalize and (np.linalg.norm(result) != 0):
            return result / np.linalg.norm(result)
        else:
            return result
        
    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        :param X: array of texts to transform
        :return: array of transformed texts
        """
        assert self.idf is not None
        return np.stack([self._text_to_tf_idf(text) for text in X])