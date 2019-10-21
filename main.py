from sklearn.cluster import KMeans
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


def make_pipeline(stop_words):
    return Pipeline([
        ('vect', CountVectorizer(stop_words=stop_words)),
        ('tfidf', TfidfTransformer()),
        ('cls', KMeans(n_clusters=50))
    ])


def collect_results(raw_lines, clustered):
    results = {a: [] for a in range(50)}
    for [line, cluster] in zip(raw_lines, clustered):
        results.get(cluster).append(line)
    results_dir = "results_" + datetime.datetime.today().isoformat()
    os.mkdir(results_dir)
    for cluster_number, cluster_lines in results.items():
        with open(results_dir + "/" + str(cluster_number) + ".txt", "w+") as f:
            for row in cluster_lines:
                f.write(row + "\n")


def run():
    [ids, raw_lines] = read_raw_data()
    data = np.array(raw_lines)

    stop_words = read_stop_words()

    pipeline = make_pipeline(stop_words)

    clustered = pipeline.fit_predict(data)

    collect_results(raw_lines, clustered)


if __name__ == '__main__':
    run()
