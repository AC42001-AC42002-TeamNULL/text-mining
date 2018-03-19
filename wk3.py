import requests
from bs4 import BeautifulSoup
import re


def get_review_texts(review_html):
    """Get all the reviews on a review page"""
    soup = BeautifulSoup(review_html, 'lxml')
    block = soup.find_all("span", class_="review-text")
    cleanr = re.compile('<.*?>')

    reviews = []

    for blk in range(0, len(block)):
        reviews.append(re.sub(cleanr, '', str(block[blk])))

    return reviews


def get_all_reviews(review_url):
    """Get all the reviews, given a review page URL"""
    reviews = []
    review_html = requests.get(review_url).text
    reviews.extend(get_review_texts(review_html))
    return reviews


url = 'https://www.amazon.com/Play-Doh-Modeling-Compound-Exclusive-Non-Toxic/product-reviews/B00JM5GW10/?pageNumber='

reviews = []
for i in range(1, 30):
    url = url + str(i)
    reviews.extend(get_all_reviews(url))

print(len(reviews), reviews[0])
