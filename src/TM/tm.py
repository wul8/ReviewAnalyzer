import gensim.downloader as api
import nltk
import numpy as np

class tm:
    """a tm object, you can do topical modeling"""
    def __init__(self, model_dir = "glove-wiki-gigaword-300"):
        self.model = api.load(model_dir) # download the corpus and return it opened as an iterable

    def similarity(self,word, category):
        """
        Generate similarity between one aspect words and all category
        :param word: str - one aspect word
        :param category: list[str] - pre defined category
        :return: list[int] - similarity
        """
        word = nltk.word_tokenize(word)
        result = [0,0,0,0]
        for w_idx in range(len(word)):
            word_vector = [0,0,0,0]
            for c_idx in range(len(category)):
                try:
                    word_vector[c_idx] =  self.model.similarity(category[c_idx],word[w_idx])
                except:
                    word_vector = [0,0,0,0]
                    break
            result = [result[i] + word_vector[i] for i in range(len(result))]
        result = [t/len(word) for t in result]
        return result

    def predict(self, words, category = None):
        """
        return the each aspect's category
        choose the category based on similarity
        :param words: list[str] - list of aspect words
        :param category: list[str] - pre-defined category. Coulda change.
        :return: list[str] - list of category
        """
        if category == None:
            category = ["food", "service", "price", "environment"]
        result = []
        for word in words:
            similarity = self.similarity(word,category)
            result.append(category[np.argmax(similarity)])
        return result




