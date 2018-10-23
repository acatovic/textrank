import numpy as np

def build_coo_matrix(sentences, word_to_ix):
    S = np.zeros((len(word_to_ix), len(word_to_ix)))

    for sent in sentences:
        for src, target in zip(sent[:-1], sent[1:]):
            if src.lower() == target.lower():
                continue
            
            S[word_to_ix[src]][word_to_ix[target]] = 1
            S[word_to_ix[target]][word_to_ix[src]] = 1
    
    return normalize_matrix(S)


def build_similarity_matrix(sentences):
    S = np.zeros((len(sentences), len(sentences)))

    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i == j:
                continue
            
            S[i][j] = sentence_similarity(sentences[i], sentences[j])
    
    return normalize_matrix(S)

def get_topk_keywords(keyword_ranks, ix_to_word, k=5):
    indexes = list(keyword_ranks.argsort())[-k:]
    return [ix_to_word[ix] for ix in indexes]

def get_topk_sentences(sentence_ranks, sentences, k=3):
    indexes = list(reversed(sentence_ranks.argsort()))[:k]
    return [sentences[i] for i in indexes]

def normalize_matrix(S):
    for i in range(len(S)):
        if S[i].sum() == 0:
            S[i] = np.ones(len(S))
        
        S[i] /= S[i].sum()
    
    return S

def sentence_similarity(sent1, sent2):
    overlap = len(set(sent1).intersection(set(sent2)))

    if overlap == 0:
        return 0
    
    return overlap / (np.log10(len(sent1)) + np.log10(len(sent2)))

def pagerank(A, eps=0.0001, d=0.85):
    R = np.ones(len(A))
    
    while True:
        r = np.ones(len(A)) * (1 - d) + d * A.T.dot(R)
        if abs(r - R).sum() <= eps:
            return r
        R = r