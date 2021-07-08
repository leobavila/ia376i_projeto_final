"""
https://github.com/castorini/pygaggle/blob/master/docs/experiments-robust04-monot5-tpu.md
https://github.com/castorini/pygaggle/blob/master/pygaggle/data/create_robust04_monot5_input.py

python ./pygaggle/data/create_robust04_monot5_input.py \
--queries = /data/robust04/topics.robust04.txt \
--run = /anserini/clean_runs/run.robust04.bm25.topics.robust04.XXXXXhits.txt \
--corpus = /data/robust04/trec_disks_4_and_5_concat.txt \
--output_segment_texts = /data/robust04/segment_textsXXXXXhits.txt \
--output_segment_query_doc_ids = /data/robust04/segment_query_doc_idsXXXXXhits.tsv

This script creates monoT5 input files by taking corpus,
queries and the retrieval run file for the queries and then
create files for monoT5 input. Each line in the monoT5 input
file follows the format:  f'Query: {query} Document: {document} Relevant:\n')
"""

import os
import re
from time import time
from pygaggle.rerank.base import Query, Text
from pygaggle.rerank.transformer import MonoT5

def main():

    DATA_DIR = "/home/l224016/projects/ia376_projeto_final/data/robust04/"
    RUNS_DIR = "/home/l224016/projects/ia376_projeto_final/data/robust04/runs"
    SEGMENTS_DIR = "/home/l224016/projects/ia376_projeto_final/data/robust04/segment_texts/"
    QUERY_DOC_DIR = "/home/l224016/projects/ia376_projeto_final/data/robust04/query_doc_ids/"
    OUTPUT_DIR = "/home/l224016/projects/ia376_projeto_final/data/robust04/monot5/"

    list_of_segment_texts = [
                            'segment_texts_1000hits.txt', 'segment_texts_2000hits.txt',
                            'segment_texts_3000hits.txt',  'segment_texts_4000hits.txt',
                            'segment_texts_5000hits.txt',  'segment_texts_6000hits.txt',
                            'segment_texts_7000hits.txt',  'segment_texts_8000hits.txt',
                            'segment_texts_9000hits.txt',  'segment_texts_10000hits.txt'
                        ]

    list_of_query_doc_ids = [
                            'segment_query_doc_ids_1000hits.tsv', 'segment_query_doc_ids_2000hits.tsv',
                            'segment_query_doc_ids_3000hits.tsv', 'segment_query_doc_ids_4000hits.tsv',
                            'segment_query_doc_ids_5000hits.tsv', 'segment_query_doc_ids_6000hits.tsv',
                            'segment_query_doc_ids_7000hits.tsv', 'segment_query_doc_ids_8000hits.tsv',
                            'segment_query_doc_ids_9000hits.tsv', 'segment_query_doc_ids_10000hits.tsv'
                        ]

   # Model
    reranker =  MonoT5()

    segment_texts = os.path.join(SEGMENTS_DIR, 'segment_texts_5000hits_001.txt')
    query_doc_ids = os.path.join(QUERY_DOC_DIR, 'segment_query_doc_ids_5000hits_001.tsv')
    monot5_results = os.path.join(OUTPUT_DIR, 'monot5_results_5000hits_001.txt')
    
    with open(segment_texts) as seg_file, open(query_doc_ids) as qdoc_file:

        passages = []

        qdoc_first_line = qdoc_file.readline()
        qdoc_first_line = qdoc_first_line.replace("\n", "").split('\t')
        query_id_old = qdoc_first_line[0]
        doc_id = qdoc_first_line[1]

        seg_first_line = seg_file.readline()
        result = re.search(r'Query\:(.*?)Document\:', seg_first_line)
        query_text = result.group(1).strip()
        result = re.search(r'Document\:(.*?)Relevant\:', seg_first_line)
        segment_text = result.group(1).strip()

        passages.append([doc_id, segment_text])

        for seg_line, qdoc_line in zip(seg_file, qdoc_file):

            qdoc_line = qdoc_line.replace("\n", "").split('\t')
            query_id = qdoc_line[0]
            doc_id = qdoc_line[1]

            if query_id == query_id_old:

                result = re.search(r'Document\:(.*?)Relevant\:', seg_line)
                segment_text = result.group(1).strip()

                passages.append([doc_id, segment_text])

            else:
                # Reranker using pygaggle
                query = Query(query_text)
                texts = [ Text(p[1], {'docid': p[0]}, 0) for p in passages]
                start = time()
                ranked_results = reranker.rerank(query, texts)
                end = time()
                time_elapsed = end - start
                print("Time Elapsed: {:.1f}".format(time_elapsed))

                # Get scores from the reranker
                final_t5_scores = {}
                for result in ranked_results:
                    if result.metadata["docid"] not in final_t5_scores:
                        final_t5_scores[result.metadata["docid"]] = result.score
                    else:
                        if final_t5_scores[result.metadata["docid"]] < result.score:
                            final_t5_scores[result.metadata["docid"]] = result.score

                # Writes a run file in the TREC format
                for rank, (docid, score) in enumerate(final_t5_scores.items()):
                    with open(monot5_results, mode='a') as writer:
                        writer.write(f'{query_id_old} Q0 {docid} {rank + 1} {1 / (rank + 1)} T5\n')
                
                # Restart variables for a new query
                passages = []

                query_id_old = query_id

                result = re.search(r'Query\:(.*?)Document\:', seg_line)
                query_text = result.group(1).strip()
                result = re.search(r'Document\:(.*?)Relevant\:', seg_line)
                segment_text = result.group(1).strip()

                passages.append([doc_id, segment_text])

        # Reranker using pygaggle
        query = Query(query_text)
        texts = [ Text(p[1], {'docid': p[0]}, 0) for p in passages]
        start = time()
        ranked_results = reranker.rerank(query, texts)
        end = time()
        time_elapsed = end - start
        print("Time Elapsed: {:.1f}".format(time_elapsed))

        # Get scores from the reranker
        final_t5_scores = {}
        for result in ranked_results:
            if result.metadata["docid"] not in final_t5_scores:
                final_t5_scores[result.metadata["docid"]] = result.score
            else:
                if final_t5_scores[result.metadata["docid"]] < result.score:
                    final_t5_scores[result.metadata["docid"]] = result.score

        # Writes a run file in the TREC format
        for rank, (docid, score) in enumerate(final_t5_scores.items()):
            with open(monot5_results, mode='a') as writer:
                writer.write(f'{query_id_old} Q0 {docid} {rank + 1} {1 / (rank + 1)} T5\n')

if __name__ == "__main__":
    main()