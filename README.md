# text-mining

## Overview

Contained is the exercises given during the AC42001-AC42002 module for text-mining.

## Installation

You will need Python==3.5 and [pipenv](https://docs.pipenv.org/) for python.

You can install most requirements by running:

```python
pipenv install
```

You will also have to download the nltk datasets by running this in a python console:

```python
import nltk

nltk.download()
```

To enter the pipenv shell run:

```python
pipenv shell
```

To execute exercises please run within the pipenv shell:

```python
python wk1.py
```

Where `wk1.py` is the file you wish to run.

## Things to note

Each file is self-contained and contains all the code for that weeks exercises. Hence a naming scheme of `wk` suffix week `1..4`.

In addition you will need to be registered for a twitter dev account to access the twitter api via tweepy. Please see [here](https://developer.twitter.com/) for more details.