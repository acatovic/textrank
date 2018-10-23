# TextRank

Simple and clean Python implementation of TextRank based on (Mihalcea and Tarau,
2004). Implements both keyword extraction, as well as extractive summarization.

## Prerequisites

- Python 2.7
- [NumPy](http://www.numpy.org/)
- [NLTK](https://www.nltk.org/)

## Usage

Extract top 10 keywords from document.txt:

```bash
python textrank.py -p document.txt -l 10
```

Summarize document.txt in 3 lines:

```bash
python textrank.py -p document.txt -s -l 3
```

## Implementation Details

TextRank is a graph-based ranking algorithm inspired by Google's PageRank (Brin
and Page, 1998). It takes into account global information computed recursively
from the entire graph. It is a __fast__, completely __unsupervised__ method
that requires __no training__. In this implementation we make the design
choices that yield the best results as per (Mihalcea and Tarau, 2004).

The following provides a summary of the design choices:

- Undirected graphs for keyword extraction
- The graphs are created using a co-occurrence matrix with co-occurrence window
N=2
- Similarity matrix for sentence extraction/summarization
- Similarity measure is based on normalized word overlap between adjacent
sentences
- Text is normalized to lower-case, and some non-interpretable unicode
characters are replaced by their proper unicode counterparts
- Tokenization is performed using NLTK's enhanced Treebank Word Tokenizer
- NLTK's built-in English stopword list is used to remove "unimportant" words
- Part-of-speech (POS) tagging is employed, and only nouns and adjectives
selected
- NLTK's Porter stemmer is used during sentence extraction/summarization but
not during keyword extraction
- For the ranking algorithm a damping factor of 0.85 is used, while the delta
rank score of 0.0001 is employed

Note that precision, recall and f1-score will not be exactly the same as in the
original paper. In the original paper no further details are given on the
syntactic filter used, besides POS tags; also different POS taggers will give
different results based on their implementation. Also it is unclear what kind
of tokenization is employed in the original paper. It is also unclear what kind
of stopwords are used. In this implementation we actually obtain better results
than in the original paper (based on Hulth and DUC datasest).

## Examples

Three-line summary of a random political article online:

> The Liberal party has wheeled out its elder statesman, former prime minister
> John Howard, in a last-ditch attempt to convince Liberal voters in Wentworth
> not to punish the government with a protest vote on the weekend. “I don’t
> think those normal Liberal voters in Wentworth want a Labor government,”
> Howard said. During a street walk in Double Bay, Howard experienced first
> hand the sentiments in Wentworth, with one voter telling him candidly he
> was appalled by the treatment of Turnbull at the hands of his own party and
> would not be voting Liberal.

The top-10 keywords based on the same article above:

> people; campaign; vote; Wentworth; Turnbull; Phelps; party; Liberal; Howard;
> government

Three-line summary of a tech article from the BBC data set (Greene and
Cunningham, 2006):

> Microsoft is investigating a trojan program that attempts to switch off the
> firm's anti-spyware software. Microsoft said it did not believe the program
> was widespread and recommended users to use an anti-virus program. Earlier
> this week, Microsoft said it would buy anti-virus software maker Sybari
> Software to improve its security in its Windows and e-mail software.

The top-10 keywords based on the previous article:

> PC; tool; anti-spyware; e-mail; anti-virus; security; software; users;
> program; Microsoft

## Future Improvements

- For keyword extraction, join adjacent top-ranked keywords in the text, i.e. if
we have _former prime minister John Howard_, and both _prime_ and _minister_
are top ranked keywords selected, we should join them into a single keyword
_prime minister_

- Use pre-trained GloVe embeddings and perform similarity and co-occurrence
weightings using GloVe vectors

- Sometimes the sentence ordering in the extracted summary is sub-optimal;
improve this using either syntax rules, or an additional model that performs
sentence ordering

- Add a capability for fetching articles/text from websites, perform HTML
parsing, and then run keyword/sentence extraction

## References

R. Mihalcea and P. Tarau. 2004. TextRank: Bringing Order into Texts.

D. Greene and P. Cunningham. 2006. Practical Solutions to the Problem of
Diagonal Dominance in Kernel Document Clustering. In _Proc. 23rd 
International Conference on Machine learning (ICML'06)_. ACM Press.

S. Brin and L. Page. 1998. The anatomy of large-scale hyper-textual Web search
engine. _Computer Networks and ISDN Systems_, 30(1-7)