import logging
import time
from random import sample, choices
from typing import List, Union, NamedTuple, Set

import nltk
import requests as requests
from gensim import models

SCORE_API = "https://cemantle.herokuapp.com/score"


class WordScore(NamedTuple):
    """So that w[1] = w.score and w[0] = w.word."""
    word: str
    score: float

    def __repr__(self):
        return f"{self.word}: {self.score}"


class SimilarityModel:
    def __init__(self):
        nltk.download('word2vec_sample')
        model_name = str(nltk.find('models/word2vec_sample/pruned.word2vec.txt'))
        self.model = models.KeyedVectors.load_word2vec_format(model_name)
        self._random_words = None

    def get_similar(self, words, not_words=None, n=1):
        not_words = not_words or []
        if not isinstance(words, List):
            words = [words]
        if not isinstance(not_words, List):
            not_words = [not_words]
        try:
            similars = self.model.most_similar(positive=words, negative=not_words, topn=n)
        except KeyError as e:
            # Ignore WTF error: KeyError: "Key 'cubism' not present in vocabulary"
            logging.info(e)
            return []

        logging.info(f"Similar to {words} and NOT {not_words}: {similars}")
        return [str.lower(word) for word, score in similars]

    def get_random_word(self):
        """Get large sample once but return values one by one"""
        if not self._random_words:
            self._random_words = sample(self.model.index_to_key, 300)
        return str.lower(self._random_words.pop(0))


class Cemantix():
    def __init__(self, similarity_model: SimilarityModel):
        self._history: Set[WordScore] = set()
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
        return [w.score ** power for w in self.history]

    def show(self, n=20):
        logging.info(f"Scores {self.iter}: {self.history[:n]}")

    def get_score(self, word: Union[str, List[str]]):
        if isinstance(word, List):
            for each_word in word:
                self.get_score(each_word)
            return

        if word in self.already_tried:
            logging.info(f"Skip {word} already tried")
            return
        self.already_tried.append(word)

        time.sleep(0.2)
        score_response = score_api(word).json()
        if "error" in score_response:
            logging.warning(f"{word} word unknown.")
            return

        score: float = score_response["score"]
        self._history.add(WordScore(word, round(100 * score)))
        self.show()
        self.iter += 1
        if score == 1:
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
        return choices(
            population=self.words,
            weights=self.scores(power=3),
            k=1
        )

    def increment(self):
        similar_words = self.w2vec.get_similar(
            [self.history[0].word, self.w2vec.get_random_word()], n=3)
        self.get_score(similar_words)

    def win(self, word):
        logging.INFO(f"Goooooooooooaaal: <<<      {word}      >>>")
        exit()


def score_api(word):
    if not word or word == "":
        raise ValueError("Empty word")
    response = requests.post(SCORE_API, data={"word": str.lower(word)})
    return response


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    game = Cemantix(SimilarityModel())

    game.play_random_words(1)
    while True:
        game.increment()
