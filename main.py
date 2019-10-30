from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from stop_words import get_stop_words
import numpy as np
import json
import os
import datetime

def read_stop_words():
    with open('stopwords.json', 'r') as data_file:
        json_data = data_file.read()

    return get_stop_words('russian') + json.loads(json_data)


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


def make_pipelines():
    return [
        Pipeline([
            ('vect', CountVectorizer(stop_words=stop_words)),
            ('tfidf', TfidfTransformer()),
            ('cls', KMeans(n_clusters=50))
        ]),
        Pipeline([
            ('vect', CountVectorizer(stop_words=stop_words)),
            ('tfidf', TfidfTransformer()),
            ('cls', SpectralClustering(n_clusters=50))
        ])
    ]


def collect_results():
    for clustered in clusteredResults:
        results_dir = "results_" + datetime.datetime.today().isoformat()
        os.mkdir(results_dir)
        with open(results_dir + "/" + "clustered.data", "w+") as f:
            for [id, cluster] in zip(ids, clustered):
                f.write(str(cluster) + ":" + str(id) + "\n")


if __name__ == '__main__':
    [ids, raw_lines] = read_raw_data()
    data = np.array(raw_lines)

    stop_words = read_stop_words()

    pipelines = make_pipelines()

    clusteredResults = []

    for pipeline in pipelines:
        clusteredResults.append(pipeline.fit_predict(data))

    collect_results()
