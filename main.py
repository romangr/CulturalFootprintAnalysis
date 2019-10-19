from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import json


def extract_id_and_text(json_object):
    json_dict = json.loads(json_object)
    return [json_dict['source']['tweetId'], json_dict['data']]


def read_raw_data():
    raw_data_lines = []
    raw_data_ids = []
    with open("raw.data") as f:
        for line in f:
            if "{" in line:
                id_data = extract_id_and_text(line)
                raw_data_ids.append(id_data[0])
                raw_data_lines.append(id_data[1])
    return [raw_data_ids, raw_data_lines]


def make_pipeline():
    return Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('cls', KMeans(n_clusters=2))
    ])


if __name__ == '__main__':
    [ids, raw_lines] = read_raw_data()
    data = np.array(raw_lines)

    pipeline = make_pipeline()

    clustered = pipeline.fit_predict(data)
    print(raw_lines[:10])
    print(clustered[:10])
