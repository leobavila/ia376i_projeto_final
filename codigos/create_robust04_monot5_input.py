"""
This script creates monoT5 input files by taking corpus,
queries and the retrieval run file for the queries and then
create files for monoT5 input. Each line in the monoT5 input
file follows the format:
    f'Query: {query} Document: {document} Relevant:\n')
"""

import os
import re
import spacy
import collections
from tqdm import tqdm

def load_corpus(path):
    print('Loading corpus...')
    corpus = {}
    n_text = 0
    with open(path, errors='ignore') as f:
        all_text = ' '.join(f.read().split())
        for raw_text in tqdm(all_text.split('</DOC>')):
            if not raw_text:
                continue
            result = re.search(r'\<DOCNO\>(.*)\<\/DOCNO\>', raw_text)
            if not result:
                continue
            doc_id = result.group(1)
            doc_id = doc_id.strip()

            result = re.search(r'\<TEXT\>(.*)\<\/TEXT\>', raw_text)
            doc_text = ''
            if result:
                doc_text = result.group(1)
                doc_text = doc_text.replace('<P>', ' ').replace('</P>', ' ')
                doc_text = doc_text.strip()
                if doc_text:
                    n_text += 1

            corpus[doc_id] = ' '.join(doc_text.split())

    print(f'Loaded {len(corpus)} docs, {n_text} with texts.')
    return corpus

def load_queries(path):
    """Loads queries into a dict of key: query_id, value: query text."""
    print('Loading queries...')
    queries = {}
    with open(path) as f:
        all_text = ' '.join(f.read().split())

        for query_text in tqdm(all_text.split('</top>')):
            if not query_text:
                continue
            result = re.search(r'\<num\>(.*)\<title\>', query_text)
            query_id = result.group(1)
            query_id = query_id.replace('Number: ', '')
            query_id = ' '.join(query_id.split())

            result = re.search(r'\<title\>(.*)\<desc\>', query_text)
            title = result.group(1)
            title = title.strip()

            result = re.search(r'\<desc\>(.*)\<narr\>', query_text)
            desc = result.group(1)
            desc = desc.replace('Description:', '')
            desc = desc.strip()

            result = re.search(r'\<narr\>(.*)', query_text)
            narr = result.group(1)
            narr = narr.replace('Narrative:', '')
            narr = narr.strip()

            query_text = desc
            if not query_text:
                query_text = title
            queries[query_id] = query_text
    return queries

def load_run(path):
    """Loads run into a dict of key: query_id, value: list of candidate doc
    ids."""
    print('Loading run...')
    run = collections.OrderedDict()
    with open(path) as f:
        for line in tqdm(f):
            query_id, _, doc_title, rank, _, _ = line.split(' ')
            if query_id not in run:
                run[query_id] = []
            run[query_id].append((doc_title, int(rank)))

    # Sort candidate docs by rank.
    sorted_run = collections.OrderedDict()
    for query_id, doc_titles_ranks in run.items():
        sorted(doc_titles_ranks, key=lambda x: x[1])
        doc_titles = [doc_titles for doc_titles, _ in doc_titles_ranks]
        sorted_run[query_id] = doc_titles

    return sorted_run

if __name__ == "__main__":

    DATA_DIR = "/home/l224016/projects/ia376_projeto_final/data/robust04/"
    RUNS_DIR = "/home/l224016/projects/ia376_projeto_final/data/robust04/runs"
    SEGMENTS_DIR = "/home/l224016/projects/ia376_projeto_final/data/robust04/segment_texts/"
    QUERY_DOC_DIR = "/home/l224016/projects/ia376_projeto_final/data/robust04/query_doc_ids/"
    OUTPUT_DIR = "/home/l224016/projects/ia376_projeto_final/data/robust04/monot5/"

    queries_path = os.path.join(DATA_DIR, "topics.robust04.txt")
    corpus_path = os.path.join(DATA_DIR, "trec_disks_4_and_5_concat.txt")
    max_length = 10
    stride = 5

    #list_of_files = []
    #for file in os.listdir(RUNS_DIR):
    #    if file.endswith(".txt"):
    #        list_of_files.append(file)

    list_of_files = ['run.robust04.bm25.topics.robust04.1000hits.txt',
                    'run.robust04.bm25.topics.robust04.2000hits.txt',
                    'run.robust04.bm25.topics.robust04.3000hits.txt',
                    'run.robust04.bm25.topics.robust04.4000hits.txt',
                    'run.robust04.bm25.topics.robust04.5000hits.txt',
                    'run.robust04.bm25.topics.robust04.6000hits.txt',
                    'run.robust04.bm25.topics.robust04.7000hits.txt',
                    'run.robust04.bm25.topics.robust04.8000hits.txt',
                    'run.robust04.bm25.topics.robust04.9000hits.txt',
                    'run.robust04.bm25.topics.robust04.10000hits.txt']

    queries = load_queries(path=queries_path)
    corpus = load_corpus(path=corpus_path)

    for run_file in list_of_files:
        hits = re.search(r'[0-9]+hits', run_file).group(0)
        runs_path = os.path.join(RUNS_DIR, f"run.robust04.bm25.topics.robust04.{hits}.txt")

        output_segment_texts_path = os.path.join(SEGMENTS_DIR, f"segment_texts_{hits}.txt")
        output_segment_query_doc_ids_path = os.path.join(QUERY_DOC_DIR, f"segment_query_doc_ids_{hits}.tsv")

        nlp = spacy.blank("en")
        nlp.add_pipe('sentencizer')
        run = load_run(path=runs_path)

        n_segments = 0
        n_docs = 0
        n_doc_ids_not_found = 0
        print(f'Converting {runs_path} to segments...')
        with open(output_segment_texts_path, 'w') as fout_segment_texts, \
                open(output_segment_query_doc_ids_path,
                    'w') as fout_segment_query_doc_ids:

            for query_id, doc_ids in tqdm(run.items(), total=len(run)):
                query_text = queries[query_id]
                for doc_id in doc_ids:
                    if doc_id not in corpus:
                        n_doc_ids_not_found += 1
                        continue
                    n_docs += 1
                    doc_text = corpus[doc_id]
                    doc = nlp(doc_text[:10000])
                    sentences = [str(sent).strip() for sent in doc.sents]
                    for i in range(0, len(sentences), stride):
                        segment = ' '.join(sentences[i:i + max_length])
                        fout_segment_query_doc_ids.write(f'{query_id}\t{doc_id}\n')
                        fout_segment_texts.write(
                            f'Query: {query_text} Document: {segment} Relevant:\n')
                        n_segments += 1
                        if i + max_length >= len(sentences):
                            break

        print(f'Wrote {n_segments} segments from {n_docs} docs.')
        print(f'{n_doc_ids_not_found} doc ids not found in the corpus.')
        print(f'{runs_path} Done!')