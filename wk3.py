import requests
from bs4 import BeautifulSoup
import re
from nltk import pos_tag, word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment import SentimentAnalyzer


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


## Retrieve reviews from amazon

url = 'https://www.amazon.com/Play-Doh-Modeling-Compound-Exclusive-Non-Toxic/product-reviews/B00JM5GW10/?pageNumber='

reviews = []
for i in range(1, 2):
    url = url + str(i)
    reviews.extend(get_all_reviews(url))

# Create sentiment analyser
sentim_analyzer = SentimentAnalyzer()
sid = SentimentIntensityAnalyzer()

segements = []
segementsPol = []
rule = 'VBD'

# POS tag our reviews by words
for review in reviews:
    tokens = word_tokenize(review)
    pos_tags = pos_tag(tokens)

    # Extract and split segments based on VBD tagging
    for tag in pos_tags:
        if tag[1] == rule:  # split based on verb, past tense
            segements.extend(review.split(tag[0]))

# Determine the segment polarity using VADER
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

p_pos = None
c_pos = None
c_guess = 0
i_guess = 0
# For each segment try to see if there's an indicator for the proceeding segments based on rule VBD markers
# The idea being that if you split an argument into two the connecting discourse marker in this case the verb will
# split an argument into premise/prefix and explanation/suffix and so will have matching polarities
for seg, pos in segementsPol:
    if p_pos is None:
        p_pos = pos

    c_pos = pos

    if c_pos == p_pos:
        c_guess += 1
    i_guess += 1

print('Using ' + rule + '. We correctly matched argument segments in ' + str(c_guess) + ' out of ' + str(
    i_guess) + ' segments')

# print(pos_tags)

# for review in reviews:
#     tok = word_tokenize(review)
#     tags = pos_tag(tok)
#     owners_possessions = []
#     for i in tags:
#         if i[1] == "POS":
#             owner = i[0])
#             possession = i[1]
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
