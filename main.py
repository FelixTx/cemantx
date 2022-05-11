import nltk
import requests as requests
from gensim.models import KeyedVectors
from nltk import find

SCORE_API = "https://cemantle.herokuapp.com/score"
HISTORY = {}


def nltk_sample():
    nltk.download('word2vec_sample')
    word2vec_sample = str(find('models/word2vec_sample/pruned.word2vec.txt'))
    return word2vec_sample


def get_model():
    word2vec_sample = nltk_sample()
    model = KeyedVectors.load_word2vec_format(word2vec_sample, binary=False)
    return model


def score_api(word):
    response = requests.post(SCORE_API, data={"word": word})
    return response


def get_score(word: str):
    score_response = score_api(word)
    score = score_response.json()["score"]
    HISTORY[word] = score
    print("Cemantle scores: ", HISTORY)
    return score


def get_similar(model, word):
    return model.most_similar(positive=[word], negative=["nothing"], topn=1)[0]


if __name__ == '__main__':
    word_1 = "first"
    get_score(word_1)

    model = get_model()
    similar_word, similarity = get_similar(model, word_1)
    print("Word2Vec similar word: ", similar_word, similarity)

    get_score(similar_word)
