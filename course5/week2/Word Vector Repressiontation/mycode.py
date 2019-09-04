import numpy as np
from w2v_utils import *

words, word_to_vec_map = read_glove_vecs('data/glove.6B.50d.txt')

def cosine_similarity(u, v):
    """
    Cosine similarity reflects the degree of similariy between u and v

    Arguments:
        u -- a word vector of shape (n,)
        v -- a word vector of shape (n,)

    Returns:
        cosine_similarity -- the cosine similarity between u and v defined by the formula above.
    """

    uu = np.sqrt(np.sum(np.square(u)))
    vv = np.sqrt(np.sum(np.square(v)))
    dot = np.sum(np.multiply(u, v))
    cos = dot / (uu * vv)
    return cos

def test_cosine_similarity():
    father = word_to_vec_map["father"]
    mother = word_to_vec_map["mother"]
    ball = word_to_vec_map["ball"]
    crocodile = word_to_vec_map["crocodile"]
    france = word_to_vec_map["france"]
    italy = word_to_vec_map["italy"]
    paris = word_to_vec_map["paris"]
    rome = word_to_vec_map["rome"]

    print("cosine_similarity(father, mother) = ", cosine_similarity(father, mother))
    print("cosine_similarity(ball, crocodile) = ",cosine_similarity(ball, crocodile))
    print("cosine_similarity(france - paris, rome - italy) = ",cosine_similarity(france - paris, rome - italy))

def complete_analogy(word_a, word_b, word_c, word_to_vec_map):
    """
    Performs the word analogy task as explained above: a is to b as c is to ____.

    Arguments:
    word_a -- a word, string
    word_b -- a word, string
    word_c -- a word, string
    word_to_vec_map -- dictionary that maps words to their corresponding vectors.

    Returns:
    best_word --  the word such that v_b - v_a is close to v_best_word - v_c, as measured by cosine similarity
    """
    # convert words to lower case
    word_a, word_b, word_c = word_a.lower(), word_b.lower(), word_c.lower()
    word_d = word_c

    e_a = word_to_vec_map[word_a]
    e_b = word_to_vec_map[word_b]
    e_c = word_to_vec_map[word_c]
    e_d = e_c
    cos_d = -1

    for w_x, e_x in word_to_vec_map.items():
        if w_x == word_a or w_x == word_b or w_x == word_c:
            continue
        cos_x = cosine_similarity(e_a - e_b, e_c - e_x)
        if cos_x > cos_d:
            cos_x = cos_d
            word_d = w_x
            e_d = e_x

    return word_d


def neutralize(word, g, word_to_vec_map):
    """
    Removes the bias of "word" by projecting it on the space orthogonal to the bias axis.
    This function ensures that gender neutral words are zero in the gender subspace.

    Arguments:
        word -- string indicating the word to debias
        g -- numpy-array of shape (50,), corresponding to the bias axis (such as gender)
        word_to_vec_map -- dictionary mapping words to their corresponding vectors.

    Returns:
        e_debiased -- neutralized word vector representation of the input "word"
    """
    e = word_to_vec_map[word]
    e_biascomponent = np.dot(e, g) / np.sum(np.square(g)) * g
    return e - e_biascomponent

def equalize(pair, bias_axis, word_to_vec_map):
    """
    Debias gender specific words by following the equalize method described in the figure above.

    Arguments:
    pair -- pair of strings of gender specific words to debias, e.g. ("actress", "actor")
    bias_axis -- numpy-array of shape (50,), vector corresponding to the bias axis, e.g. gender
    word_to_vec_map -- dictionary mapping words to their corresponding vectors

    Returns
    e_1 -- word vector corresponding to the first word
    e_2 -- word vector corresponding to the second word
    """
    g = bias_axis

    word1, word2 = pair
    e_w1, e_w2 = word_to_vec_map[word1], word_to_vec_map[word2]
    u = (e_w1 + e_w2) / 2
    u_B = np.dot(u, g) / np.sum(np.square(g)) * g
    u_orth = u - u_B
    e_w1B = np.dot(e_w1, g) / np.sum(np.square(g)) * g
    e_w2B = np.dot(e_w2, g) / np.sum(np.square(g)) * g
    temp = np.sqrt(np.abs(1 - np.sum(np.square(u_orth))))
    e_w1B_corrected = temp * (e_w1B - u_B) / np.linalg.norm(e_w1 - u_orth - u_B)
    e_w2B_corrected = temp * (e_w2B - u_B) / np.linalg.norm(e_w2 - u_orth - u_B)
    e_1 = e_w1B_corrected + u_orth
    e_2 = e_w2B_corrected + u_orth

    return e_1, e_2


