import json
import re
import numpy
from numpy import genfromtxt
from sklearn.feature_extraction.text import CountVectorizer


def get_train_and_test(data, ratio=0.8):
    temp_data = data.copy()
    numpy.random.shuffle(temp_data)
    return temp_data[:int(ratio * len(temp_data))], temp_data[int(ratio * len(temp_data)):]


def get_X_y(data):
    vectorizer = CountVectorizer(analyzer="word",
                                 tokenizer=None,
                                 preprocessor=None,
                                 stop_words=None,
                                 token_pattern='\\b\\w+\\b')
    docs = [x[0] for x in data]
    vectorizer.fit(docs)
    integers_from_strings = [[vectorizer.vocabulary_.get(y) for y in re.sub(r'[.!,;?]', ' ', x).split() if
                              vectorizer.vocabulary_.get(y) is not None] for x in docs]
    return numpy.array(integers_from_strings), numpy.array([x[1] for x in data])


def get_sentiment_data(train_test_ratio=0.8):
    sentiment_amazon = genfromtxt('./datasets/sentiment/amazon_cells_labelled.txt', delimiter='\t', encoding="utf-8",
                                  dtype=None)
    train_amazon, test_amazon = get_train_and_test(sentiment_amazon, train_test_ratio)
    X_train_amazon, y_train_amazon = get_X_y(train_amazon)
    X_test_amazon, y_test_amazon = get_X_y(test_amazon)

    sentiment_imdb = genfromtxt('./datasets/sentiment/imdb_labelled.txt', delimiter='\t', encoding="utf-8", dtype=None)
    train_imdb, test_imdb = get_train_and_test(sentiment_imdb, train_test_ratio)
    X_train_imdb, y_train_imdb = get_X_y(train_imdb)
    X_test_imdb, y_test_imdb = get_X_y(test_imdb)

    sentiment_yelp = genfromtxt('./datasets/sentiment/yelp_labelled.txt', delimiter='\t', encoding="utf-8", dtype=None)
    train_yelp, test_yelp = get_train_and_test(sentiment_yelp, train_test_ratio)
    X_train_yelp, y_train_yelp = get_X_y(train_yelp)
    X_test_yelp, y_test_yelp = get_X_y(test_yelp)

    X_train, y_train = numpy.concatenate((X_train_amazon, X_train_imdb, X_train_yelp)), numpy.concatenate(
        (y_train_amazon, y_train_imdb, y_train_yelp)),
    X_test, y_test = numpy.concatenate((X_test_amazon, X_test_imdb, X_test_yelp)), numpy.concatenate(
        (y_test_amazon, y_test_imdb, y_test_yelp)),

    return (X_train, y_train), (X_test, y_test)


def get_troll_data(train_test_ratio=0.8):
    troll_data = []
    with open('./datasets/trollDetection.json', 'r') as f:
        json_lines = f.readlines()
        for line in json_lines:
            json_line = json.loads(line)
            troll_data.append((json_line['content'], int(json_line['annotation']['label'][0])))

    train, test = get_train_and_test(troll_data, train_test_ratio)
    X_train, y_train = get_X_y(train)
    X_test, y_test = get_X_y(test)

    return (X_train, y_train), (X_test, y_test)


def get_valley_data(train_test_ratio=0.8):
    valley_data = []
    with open('./datasets/sentiment/valley_data.txt', 'r') as f:
        file_content = f.readlines()
        for line in file_content:
            values = line.split("\t")
            valley_data.append((str(values[0]), int(re.sub("\D", "", values[1]))))
    train, test = get_train_and_test(valley_data, train_test_ratio)
    X_train, y_train = get_X_y(train)
    X_test, y_test = get_X_y(test)

    return (X_train, y_train), (X_test, y_test)
