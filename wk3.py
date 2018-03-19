import requests
from bs4 import BeautifulSoup, NavigableString

def get_review_texts(review_html):
    """Get all the reviews on a review page"""
    soup = BeautifulSoup(review_html, 'lxml')
    block = soup.find_all("span", class_="review-text")
    return block


def get_all_reviews(review_url):
    """Get all the reviews, given a review page URL"""
    reviews = []
    review_html = requests.get(review_url).text
    reviews.extend(get_review_texts(review_html))
    return reviews

url = 'https://www.amazon.com/Play-Doh-Modeling-Compound-Exclusive-Non-Toxic/product-reviews/B00JM5GW10/?pageNumber='

reviews = []
for i in range(1, 20):
    url = url + str(i)
    reviews.extend(get_all_reviews(url))

print(len(reviews))