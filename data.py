import io

from nltk import pos_tag
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import porter
from nltk.tokenize import sent_tokenize

eos_tokens = set([".", "!", "?"])

pos = {
    "NN": "n",
    "NNS": "n",
    "NNP": "n",
    "JJ": "a",
    "JJR": "a",
    "JJS": "a",
}

stemmer = porter.PorterStemmer()

stop_words = set(stopwords.words("english"))
stop_words = stop_words.union(["monday", "tuesday", "wednesday", "thursday",
                               "friday", "saturday", "sunday", "january",
                               "february", "april", "may", "june", "july",
                               "august", "september", "october", "november",
                               "december", "mon", "tue", "wed", "thu", "fri",
                               "sat", "sun", "jan", "feb", "mar", "apr", "may",
                               "jun", "jul", "aug", "sep", "oct", "nov", "dec"])

tags = set(["NN", "NNS", "NNP", "JJ", "JJR", "JJS"])

def build_vocabulary(sentences):
    word_to_ix = {}
    ix_to_word = {}

    for sent in sentences:
        for word in sent:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
                ix_to_word[len(ix_to_word)] = word
    
    return word_to_ix, ix_to_word

def filter_sentences(sentences, lowercase=True, stem=True):
    norm_sents = [normalize_sentence(s, lowercase) for s in sentences]
    filtered_sents = [filter_words(sent) for sent in norm_sents]

    if stem:
        return [stem_sentence(sent) for sent in filtered_sents]
    
    return filtered_sents

def filter_words(sentence):
    filtered_sentence = []
    for word, tag in pos_tag(word_tokenize(sentence)):
        if tag not in tags:
            continue
        
        if word.lower() in stop_words:
            continue
        
        filtered_sentence.append(word)
    
    return filtered_sentence

def is_heading(s):
    if s[-1] in eos_tokens:
        return False
    
    return True

def load_data(path):
    sentences = []
    with io.open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not is_heading(line):
                for sent in sent_tokenize(line):
                    sentences.append(sent)
    
    return sentences

def normalize_sentence(sentence, lowercase=True):
    if lowercase:
        sentence = sentence.lower()
    
    return sentence.replace(u"\u2013", u"-").replace(
        u"\u2019", u"'").replace(u"\u201c", u"\"").replace(
        u"\u201d", u"\"")

def stem_sentence(sentence):
    return [stemmer.stem(word) for word in sentence]