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
    return client["cultural"].RawRecords


def get_clusters_collection(client):
    return client["cultural"].Clusters


def read_stop_words():
    with open('stopwords.json', 'r') as data_file:
        json_data = data_file.read()

    return get_stop_words('russian') + json.loads(json_data=json_data)


def extract_id_and_text(record):
    return [record.get('_id'), str(record.get('data'))]


def read_raw_data(raw_records_collection):
    raw_data_lines = []
    raw_data_ids = []
    week = current_week()
    records = raw_records_collection.find({"week": "2020-21"})

    for record in records:
        id_data = extract_id_and_text(record)
        raw_data_ids.append(id_data[0])
        raw_data_lines.append(id_data[1])
    return [raw_data_ids, raw_data_lines]


def current_week():
    return str(datetime.datetime.today().year) + "-" \
           + str(datetime.datetime.today().isocalendar()[1])


def make_pipeline(stop_words):
    number_of_clusters = 100
    return Pipeline([
            ('vect', CountVectorizer(stop_words=stop_words)),
            ('tfidf', TfidfTransformer()),
            ('cls', KMeans(n_clusters=int(number_of_clusters)))
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
                clusters_collection.insert({"_id": cluster_id, "week": current_week()})
                cluster_ids[cluster] = cluster_id
            cluster_id = cluster_ids[cluster]
            raw_records_collection.find_and_modify(query={'_id': id}, update={"$set": {'cluster': cluster_id}}, upsert=False, full_response=False)


def collect_results(raw_lines, clustered):
  results = {a: [] for a in range(100)}
  for [line, cluster] in zip(raw_lines, clustered[0]):
    results.get(cluster).append(line)
  results_dir = "results_" + datetime.datetime.today().isoformat()
  os.mkdir(results_dir)
  for cluster_number, cluster_lines in results.items():
    with open(results_dir + "/" + str(cluster_number) + ".txt", "w+") as f:
      for row in cluster_lines:
        f.write(row + "\n")

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

    # collect_results_to_db(mongo_client, ids, clustered_results)

    collect_results(raw_lines, clustered_results)

    print("Finished " + str(datetime.datetime.today()))


if __name__ == '__main__':
    print("Starter analysis app")
    run()
    # schedule.every(1).saturday.at("13:00").do(run)
    # while 1:
    #     schedule.run_pending()
    #     time.sleep(3600)
