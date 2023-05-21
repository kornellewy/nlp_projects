"""
source :
https://realpython.com/python-nltk-sentiment-analysis/
"""

import nltk
from pprint import pprint
from random import shuffle
from statistics import mean

# download data
nltk.download([
    "names",
    "stopwords",
    "state_union",
    "twitter_samples",
    "movie_reviews",
    "averaged_perceptron_tagger",
    "vader_lexicon",
    "punkt",
])

# Compiling
words = [w for w in nltk.corpus.state_union.words() if w.isalpha()]

# get all english stop words
stopwords = nltk.corpus.stopwords.words("english")

# remove stop words
words = [w for w in words if w.lower() not in stopwords]

# tokenization example
text = """
For some quick analysis, creating a corpus could be overkill.
If all you need is a word list,
there are simpler ways to achieve that goal."""
pprint(nltk.word_tokenize(text), width=79, compact=True)

# frequency distributions
words: list[str] = nltk.word_tokenize(text)
fd = nltk.FreqDist(words)
print(fd.most_common(3))
print(fd.tabulate(3))
lower_fd = nltk.FreqDist([w.lower() for w in fd])
print(lower_fd.most_common(3))
print(lower_fd.tabulate(3))
# You could create frequency distributions of words starting with a particular letter,
# or of a particular length, or containing certain letters.

# concordance
text = nltk.Text(nltk.corpus.state_union.words())
concordance_list = text.concordance_list("america", lines=2)
for entry in concordance_list:
    print(entry.line)

# frequency distribution for a given text
words: list[str] = nltk.word_tokenize(
    """Beautiful is better than ugly.
    Explicit is better than implicit.
    Simple is better than complex."""
)
text = nltk.Text(words)
fd = text.vocab()  # Equivalent to fd = nltk.FreqDist(words)
print(fd.tabulate(3))

# collocations Bigrams, Trigrams, Quadgrams
words = [w for w in nltk.corpus.state_union.words() if w.isalpha()]
finder = nltk.collocations.TrigramCollocationFinder.from_words(words)
print(finder.ngram_fd.most_common(2))

# Pre-Trained Sentiment Analyzer
from nltk.sentiment import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()
score = sia.polarity_scores("Wow, NLTK is really powerful!")
print(score)

tweets = [t.replace("://", "//") for t in nltk.corpus.twitter_samples.strings()]

def is_positive(tweet: str) -> bool:
    """True if tweet has positive compound sentiment, False otherwise."""
    return sia.polarity_scores(tweet)["compound"] > 0

shuffle(tweets)
for tweet in tweets[:10]:
    print(">", is_positive(tweet), tweet)

positive_review_ids = nltk.corpus.movie_reviews.fileids(categories=["pos"])
negative_review_ids = nltk.corpus.movie_reviews.fileids(categories=["neg"])
all_review_ids = positive_review_ids + negative_review_ids


def is_positive(review_id: str) -> bool:
    """True if the average of all sentence compound scores is positive."""
    text = nltk.corpus.movie_reviews.raw(review_id)
    scores = [
        sia.polarity_scores(sentence)["compound"]
        for sentence in nltk.sent_tokenize(text)
    ]
    return mean(scores) > 0

shuffle(all_review_ids)
correct = 0
for review_id in all_review_ids:
    if is_positive(review_id):
        if review_id in positive_review_ids:
            correct += 1
    else:
        if review_id in negative_review_ids:
            correct += 1

print(F"{correct / len(all_review_ids):.2%} correct")


