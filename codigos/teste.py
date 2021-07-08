import os
QUERY_DOC_DIR = "/home/l224016/projects/ia376_projeto_final/data/robust04/query_doc_ids/"
query_doc_ids = os.path.join(QUERY_DOC_DIR, 'segment_query_doc_ids_1000hits.tsv')

checker = []
with open(query_doc_ids, mode='r') as f:
    for line in f:
        qdoc_first_line = line.replace("\n", "").split('\t')
        query_id = qdoc_first_line[0]
        if query_id not in checker:
            checker.append(query_id)

checker[:2]