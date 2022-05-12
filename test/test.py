from app.main import sort, random_words

H = [("a", 1), ("b", 0), ]
H_SORTED = [("b", 0),("a", 1) ]


def test_sort_history():
    assert H_SORTED == sort(H)
    assert H_SORTED == sort(H_SORTED)


def test_random():
    assert [] == random_words()
