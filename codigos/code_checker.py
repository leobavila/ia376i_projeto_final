import os
import pandas as pd

DATA_DIR = "/home/l224016/projects/ia376_projeto_final/data/robust04/"
RUNS_DIR = "/home/l224016/projects/ia376_projeto_final/data/robust04/runs"
SEGMENTS_DIR = "/home/l224016/projects/ia376_projeto_final/data/robust04/segment_texts/"
QUERY_DOC_DIR = "/home/l224016/projects/ia376_projeto_final/data/robust04/query_doc_ids/"
OUTPUT_DIR = "/home/l224016/projects/ia376_projeto_final/data/robust04/monot5/"

segment_texts = os.path.join(SEGMENTS_DIR, 'segment_texts_2000hits.txt')
query_doc_ids = os.path.join(QUERY_DOC_DIR, 'segment_query_doc_ids_2000hits.tsv')
monot5_results = os.path.join(OUTPUT_DIR, 'monot5_results_2000hits.txt')

df = pd.read_csv(query_doc_ids, sep = '\t', names = ["query_id", "doc_id"])
df.reset_index().groupby(["query_id", "doc_id"]).count().reset_index()

