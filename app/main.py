import time
import pickle
from dataclasses import dataclass
from os.path import exists
from random import sample, choices
from typing import List, Union, Callable

import nltk
import requests as requests
from gensim import models, downloader
from gensim.models import KeyedVectors

SCORE_API = "https://cemantle.herokuapp.com/score"


@dataclass
class WordScore:
    """So that w[1] = w.score and w[0] = w.word."""

    word: str
    score: float

    def __init__(self, word, score):
        self.word = word
        self.score = score

    def __repr__(self):
        return f"{self.word}: {self.score}"


def nltk():
    nltk.download("word2vec_sample")
    model_name = str(nltk.find("models/word2vec_sample/pruned.word2vec.txt"))
    return models.KeyedVectors.load_word2vec_format(model_name)


def get_text8():
    return models.Word2Vec(downloader.load("text8")).wv


def load_model(model_loader: Callable):
    if exists("model_dump.pkl"):
        with open("model_dump.pkl", "rb") as f:
            model = pickle.load(f)
    else:
        model = model_loader()
        with open("model_dump.pkl", "wb") as f:
            pickle.dump(model, f)
    return model


class SimilarityModel:
    def __init__(self, model: KeyedVectors):
        self.vocab_size = None
        self.model = model
        self._random_words = []
        self._scope_width = 500

    def set_scope_width(self, n):
        self._scope_width = n

    def get_similar(self, words, not_words=None, n=1):
        not_words = not_words or []
        if not isinstance(words, List):
            words = [words]
        if not isinstance(not_words, List):
            not_words = [not_words]
        try:
            similars = self.model.most_similar(
                positive=words,
                negative=not_words,
                restrict_vocab=self._scope_width,
                topn=n,
            )
        except KeyError as e:
            # Ignore WTF error: KeyError: "Key 'cubism' not present in vocabulary"
            print(e)
            return []

        print(f"Similar to {words} and NOT {not_words} gives: {similars}")
        return [str.lower(word) for word, score in similars if len(word) > 2]

    def get_random_word(self):
        """Get a random word from model.

        Notes:
            - batch load a large sample and return values one by one.
            - Pick the sample among a subset of the most common words.
        """
        # reload words sample if empty or vocab size has changed
        if not self._random_words:
            self._random_words = sample(self.model.index_to_key[:500], 300)
            self._random_words = [w for w in self._random_words if len(w) > 2]

        return str.lower(self._random_words.pop(0))


class Cemantix:
    def __init__(self, similarity_model: SimilarityModel):
        self._history: List[WordScore] = []
        self.w2vec = similarity_model
        self.last_similar = None
        self.already_tried = []
        self.iter = 0

    @property
    def history(self):
        return list(sorted(self._history, key=lambda item: item.score, reverse=True))

    @property
    def words(self):
        return [w.word for w in self.history]

    def scores(self, power: int = 1):
        return [w.score**power for w in self.history]

    def show(self, n=20):
        print(f"Scores {self.iter}: {self.history[:n]}")

    def get_score(self, word: Union[str, List[str]]):
        if isinstance(word, List):
            for each_word in word:
                self.get_score(each_word)
            return

        if word in self.already_tried:
            print(f"Skip {word} already tried")
            return
        if len(word) < 3:
            print(f"Shitty word {word} skipped")
            return
        self.already_tried.append(word)

        time.sleep(0.2)
        score_response = score_api(word).json()
        if "error" in score_response:
            print(f"{word} word unknown.")
            return

        score: int = int(100 * score_response["score"])
        self._history.append(WordScore(word, score))
        self.show()
        self.iter += 1
        if score == 100:
            self.win(word)
        return score

    def play_random_words(self, n: int = 1):
        k = 0
        while k < n:
            word = self.w2vec.get_random_word()
            score = self.get_score(word)
            if score:
                k += 1

    def pick_positives(self):
        """Weighted choice over top words"""
        return choices(population=self.words[:8], weights=self.scores(power=3)[:8], k=1)

    def pick_negatives(self):
        """Weighted choice over bottom words"""
        return choices(
            population=self.words, weights=[-w for w in self.scores(power=3)], k=1
        )

    def play(self):
        max_score = self.history[0].score
        self.compute_search_scope_width()
        if max_score < 10:
            self.play_random_words()

        words = self.pick_positives()
        similar_words = self.w2vec.get_similar(words, n=3)
        self.lower_score(words)

        # If similar words were already tried, get something different
        if all(word in self.already_tried for word in similar_words):
            words.append(self.w2vec.get_random_word())
            similar_words = self.w2vec.get_similar(words, n=3)
            if all(word in self.already_tried for word in similar_words):
                self.play_random_words()
                return
        return self.get_score(similar_words)

    def win(self, word):
        print(
            rf"\o/ \o/ \o/ \o/ \o/ \o/ : --->>   {word}   <<--- \o/ \o/ \o/ \o/ \o/ \o/"
        )
        exit(1)

    def lower_score(self, words):
        for w in self._history:
            if w.word in words:
                w.score -= 0.5

    def compute_search_scope_width(self):
        max_score = self.history[0].score
        if max_score < 25 and self.iter < 100:
            self.w2vec.set_scope_width(1000)
        if max_score < 50 and self.iter < 200:
            self.w2vec.set_scope_width(2000)
        if max_score < 90:
            self.w2vec.set_scope_width(5000)
        self.w2vec.set_scope_width(30000)


def score_api(word):
    if not word or word == "":
        raise ValueError("Empty word")
    response = requests.post(SCORE_API, data={"word": str.lower(word)})
    return response


if __name__ == "__main__":
    wv = load_model(get_text8)  # delete "model_dump.pkl" before changing model

    game = Cemantix(SimilarityModel(wv))

    while game.iter < 300:
        game.play()
