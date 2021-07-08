import os

SEGMENTS_DIR = "/home/l224016/projects/ia376_projeto_final/data/robust04/segment_texts/"
QUERY_DOC_DIR = "/home/l224016/projects/ia376_projeto_final/data/robust04/query_doc_ids/"

segment_texts = os.path.join(SEGMENTS_DIR, 'segment_texts_1000hits.txt')
query_doc_ids = os.path.join(QUERY_DOC_DIR, 'segment_query_doc_ids_1000hits.tsv')

segment_texts_out = os.path.join(SEGMENTS_DIR, 'test_segment_texts_1000hits.txt')
query_doc_ids_out = os.path.join(QUERY_DOC_DIR, 'test_segment_query_doc_ids_1000hits.tsv')

with open(segment_texts) as seg_file, open(query_doc_ids) as qdoc_file:

    for i in range(21782):
        seg_line = seg_file.readline()
        qdoc_line = qdoc_file.readline()

        with open(segment_texts_out, 'a') as seg_file_out:
            seg_file_out.write(seg_line)
        
        with open(query_doc_ids_out, 'a') as qdoc_file_out:
            qdoc_file_out.write(qdoc_line)

