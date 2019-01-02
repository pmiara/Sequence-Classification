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
                                 token_pattern=r'\b\w+\b')
    docs = [x[0] for x in data]
    vectorizer.fit(docs)
    integers_from_strings = [[vectorizer.vocabulary_.get(y.lower()) for y in re.sub(r'[.!,;?]', ' ', x).split() if
                              vectorizer.vocabulary_.get(y.lower()) is not None] for x in docs]
    return numpy.array(integers_from_strings), numpy.array([x[1] for x in data])


def get_sentiment_data():
    sentiment_amazon = genfromtxt('./datasets/sentiment/amazon_cells_labelled.txt', delimiter='\t', encoding="utf-8",
                                  dtype=None)
    X_amazon, y_amazon = get_X_y(sentiment_amazon)

    sentiment_imdb = genfromtxt('./datasets/sentiment/imdb_labelled.txt', delimiter='\t', encoding="utf-8", dtype=None)
    X_imdb, y_imdb = get_X_y(sentiment_imdb)

    sentiment_yelp = genfromtxt('./datasets/sentiment/yelp_labelled.txt', delimiter='\t', encoding="utf-8", dtype=None)
    X_yelp, y_yelp = get_X_y(sentiment_yelp)

    X_data, y_data = numpy.concatenate((X_amazon, X_imdb, X_yelp)), numpy.concatenate(
        (y_amazon, y_imdb, y_yelp))

    return X_data, y_data


def get_troll_data(train_test_ratio=0.8):
    troll_data = []
    with open('./datasets/trollDetection.json', 'r') as f:
        json_lines = f.readlines()
        for line in json_lines:
            json_line = json.loads(line)
            troll_data.append((json_line['content'], int(json_line['annotation']['label'][0])))

    X_troll, y_troll = get_X_y(troll_data)
    return X_troll, y_troll



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


def get_bio_data():
    bio_data = []
    with open('./datasets/bioData.data', 'r') as f:
        lines = f.readlines()
        for line in lines:
            bio_class, _, sequence = line.replace(" ", "").replace("\n", "").split(',')
            bio_data.append((' '.join(list(sequence)), bio_class))
    return get_X_y(bio_data)
