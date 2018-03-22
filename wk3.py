import requests
from bs4 import BeautifulSoup
import re
from nltk import pos_tag, word_tokenize, RegexpParser
import nltk
from nltk.sem import relextract
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment import SentimentAnalyzer
from nltk.probability import FreqDist
from itertools import chain
from nltk.classify import NaiveBayesClassifier, accuracy
import collections
from nltk.metrics.scores import precision, f_measure, recall

def get_review_texts(review_html):
    soup = BeautifulSoup(review_html, 'lxml')
    block = soup.find_all('span', class_='review-text')
    cleanr = re.compile('<.*?>')

    reviews = []

    for blk in range(0, len(block)):
        reviews.append(re.sub(cleanr, '', str(block[blk])))

    return reviews


def get_all_reviews(review_url):
    reviews = []
    review_html = requests.get(review_url).text
    reviews.extend(get_review_texts(review_html))
    return reviews


url = 'https://www.amazon.com/Play-Doh-Modeling-Compound-Exclusive-Non-Toxic/product-reviews/B00JM5GW10/?pageNumber='

reviews = []
for i in range(1, 2):
    url = url + str(i)
    reviews.extend(get_all_reviews(url))


sentim_analyzer = SentimentAnalyzer()
sid = SentimentIntensityAnalyzer()

for review in reviews:
    tokens = word_tokenize(review)
    pos_tags = pos_tag(tokens)

    segements = []
    segementsPol = []

    for tag in pos_tags:
        if tag[1] == 'VBD':  # split based on verb, past tense
            segements.extend(review.split(tag[0]))

    for segment in segements:
        ss = sid.polarity_scores(segment)

        label = ''

        if ss['compound'] >= .2:
            label = 'pos'

        if ss['compound'] <= -.2:
            label = 'neg'

        if -.2 < ss['compound'] < .2:
            label = 'neu'

        segementsPol.append([segment, label])

    print(segementsPol)

    # print(pos_tags)

# for review in reviews:
#     tok = word_tokenize(review)
#     tags = pos_tag(tok)
#     owners_possessions = []
#     for i in tags:
#         if i[1] == "POS":
#             owner = i[0].nbor(-1)
#             possession = i[0].nbor(1)
#             owners_possessions.append((owner, possession))
#     print(owners_possessions)

# grammar = 'NP: {<DT>?<JJ>*<NNP>}'
# cp = RegexpParser(grammar)

# pattern = re.compile(r'is')
#
# for review in reviews:
#     tok = word_tokenize(review)
#     tags = pos_tag(tok)
#     for rel in relextract.extract_rels('ORGANIZATION', '', tags, corpus='ace', pattern=pattern):
#         print(relextract.rtuple(rel))
