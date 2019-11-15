import datetime
import json
import os

import numpy as np
from pymongo import MongoClient
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from stop_words import get_stop_words


def get_collection():
    client = MongoClient(
        host=os.environ['MONGO_HOST'],
        port=int(os.environ['MONGO_PORT']),
        username=os.environ['MONGO_USERNAME'],
        password=os.environ['MONGO_PASSWORD']
    )
    return client[os.environ['MONGO_DATABASE']].RawRecords


def read_stop_words():
    with open('stopwords.json', 'r') as data_file:
        json_data = data_file.read()

    return get_stop_words('russian') + json.loads(json_data)


def extract_id_and_text(record):
    return [record.get('source').get("tweetId"), str(record.get('data'))]


def read_raw_data(raw_records_collection):
    raw_data_lines = []
    raw_data_ids = []
    records = raw_records_collection.find()

    for record in records:
        id_data = extract_id_and_text(record)
        raw_data_ids.append(id_data[0])
        raw_data_lines.append(id_data[1])
    return [raw_data_ids, raw_data_lines]


def make_pipelines(stop_words):
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


def collect_results(ids, clusteredResults):
    for clustered in clusteredResults:
        results_dir = "results_" + datetime.datetime.today().isoformat()
        os.mkdir(results_dir)
        with open(results_dir + "/" + "clustered.data", "w+") as f:
            for [id, cluster] in zip(ids, clustered):
                f.write(str(cluster) + ":" + str(id) + "\n")


def run():
    print("Started " + str(datetime.datetime.today()))
    rawRecordsCollection = get_collection()

    [ids, raw_lines] = read_raw_data(rawRecordsCollection)

    print("Read {} lines".format(len(raw_lines)))

    data = np.array(raw_lines)

    stop_words = read_stop_words()

    pipelines = make_pipelines(stop_words)

    clusteredResults = []

    for pipeline in pipelines:
        clusteredResults.append(pipeline.fit_predict(data))

    collect_results(ids, clusteredResults)

    print("Finished " + str(datetime.datetime.today()))


if __name__ == '__main__':
    run()
