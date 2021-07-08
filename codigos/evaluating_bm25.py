import pandas as pd
import re
import os

from tqdm import tqdm
from bs4 import BeautifulSoup

DATA_DIR = "/home/l224016/projects/ia376_projeto_final/data/robust04/"
RUNS_DIR = "/home/l224016/projects/ia376_projeto_final/anserini/clean_runs/"
OUTPUT_DIR = "/home/l224016/projects/ia376_projeto_final/data/robust04/monot5/"

corpus_path = os.path.join(DATA_DIR, "trec_disks_4_and_5_concat.txt")
runs_path = os.path.join(RUNS_DIR, f"run.robust04.bm25.topics.robust04.50000hits.txt")
monot5_results = os.path.join(OUTPUT_DIR, 'monot5_results_1000hits.txt')

# Avaliação dos documentos no arquivo "trec_disks_4_and_5_concat.txt"
def load_corpus(path):
    print('Loading corpus...')
    corpus = {}
    corpus_counting = []
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
            corpus_counting.append(doc_id)

    print(f'Loaded {len(corpus)} docs, {n_text} with texts.')
    return corpus, corpus_counting

def cleaning_tags(text):
  return BeautifulSoup(text, "lxml").text

corpus, corpus_counting = load_corpus(path=corpus_path)

df = pd.DataFrame(corpus_counting, columns = ['docs'])
df["doc_length"] = df.docs.map(lambda x: len(x))

counting_docs = df.groupby('docs').count().reset_index()
doc_counting = counting_docs.sort_values(by = "doc_length", ascending = False)
doc_counting.rename(columns = {'doc_length': 'doc_count'}, inplace = True)
doc_counting
df[df["doc_length"] > 16] # FT911-1, FT932-6862, FR940104-1-00001, FR941028-2-00001, FR940513-1-00001

df.iloc[192673].docs
df.iloc[224463].docs
df.iloc[224464].docs
df.iloc[246695].docs

# investigar se realmente sao duplicados

df[df.docs == 'FT911-1'] # documento duplicado
df[df.docs.str.contains('FT932-6862')] # nome do documento errado
df[df.docs.str.contains('FR940104-1-00001')] # nome do documento errado e duplicado
df[df.docs.str.contains('FR941028-2-00001')] # nome do documento errado
df[df.docs.str.contains('FR940513-1-00001')] # nome do documento errado

# Avaliação dos arquivos run para ver se um mesmo documento foi retornado mais de uma vez no candidate_texts de uma query
bm25 = pd.read_csv(monot5_results, sep = " ", names = ['id', 'Q0', 'docid', 'rank', 'score', 'model'])
bm25["doc_length"] = bm25.docid.map(lambda x: len(x))
bm25[["id", "docid", "rank"]].groupby(["id", "docid"]).count().reset_index().sort_values(by = "rank", ascending = False)
bm25[bm25["id"] == 301]
len(bm25["id"].unique())
bm25.groupby("")

# Avaliacao
# tools/eval/trec_eval.9.0.4/trec_eval -m map -m P.30 src/main/resources/topics-and-qrels/qrels.robust04.txt runs/run.robust04.bm25.topics.robust04.unique.9000hits.txt