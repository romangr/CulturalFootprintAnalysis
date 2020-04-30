import datetime
import json
import os
import uuid

import numpy as np
from pymongo import MongoClient
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from stop_words import get_stop_words


def get_raw_records_collection(client):
    return os.environ['MONGO_DATABASE'].RawRecords


def get_clusters_collection(client):
    return os.environ['MONGO_DATABASE'].Clusters


def read_stop_words():
    with open('stopwords.json', 'r') as data_file:
        json_data = data_file.read()

    return get_stop_words('russian') + json.loads(json_data)


def extract_id_and_text(record):
    return [record.get('_id'), str(record.get('data'))]


def read_raw_data(raw_records_collection):
    raw_data_lines = []
    raw_data_ids = []
    records = raw_records_collection.find()

    for record in records:
        id_data = extract_id_and_text(record)
        raw_data_ids.append(id_data[0])
        raw_data_lines.append(id_data[1])
    return [raw_data_ids, raw_data_lines]


def make_pipeline(stop_words):
    return Pipeline([
            ('vect', CountVectorizer(stop_words=stop_words)),
            ('tfidf', TfidfTransformer()),
            ('cls', KMeans(n_clusters=50))
        ])


def collect_results_to_files(ids, clustered_results):
    for clustered in clustered_results:
        results_dir = "results_" + datetime.datetime.today().isoformat()
        os.mkdir(results_dir)
        with open(results_dir + "/" + "clustered.data", "w+") as f:
            for [id, cluster] in zip(ids, clustered):
                f.write(str(cluster) + ":" + str(id) + "\n")


def collect_results_to_db(mongo_client, ids, clustered_results):
    raw_records_collection = get_raw_records_collection(mongo_client)
    clusters_collection = get_clusters_collection(mongo_client)
    cluster_ids = {}
    for clustered in clustered_results:
        for [id, cluster] in zip(ids, clustered):
            if cluster not in cluster_ids:
                cluster_id = uuid.uuid4()
                clusters_collection.insert({"_id": cluster_id})
                cluster_ids[cluster] = cluster_id
            cluster_id = cluster_ids[cluster]
            raw_records_collection.find_and_modify(query={'_id': id}, update={"$set": {'cluster': cluster_id}}, upsert=False, full_response=False)


def run():
    print("Started " + str(datetime.datetime.today()))
    mongo_client = MongoClient(
                           host=os.environ['MONGO_HOST'],
                           port=int(os.environ['MONGO_PORT']),
                           username=os.environ['MONGO_USERNAME'],
                           password=os.environ['MONGO_PASSWORD']
                       )
    raw_records_collection = get_raw_records_collection(mongo_client)

    [ids, raw_lines] = read_raw_data(raw_records_collection)

    print("Read {} lines".format(len(raw_lines)))

    data = np.array(raw_lines)

    stop_words = read_stop_words()

    pipeline = make_pipeline(stop_words)

    clustered_results = []

    clustered_results.append(pipeline.fit_predict(data))

    # collect_results_to_files(ids, clustered_results)
    collect_results_to_db(mongo_client, ids, clustered_results)

    print("Finished " + str(datetime.datetime.today()))


if __name__ == '__main__':
    run()
