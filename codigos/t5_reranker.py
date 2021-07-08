import os
import sys
import glob
import jsonlines

import re
import spacy
from typing import List

import torch
from tqdm import tqdm
from pygaggle.model import CachedT5ModelLoader, T5BatchTokenizer
from pygaggle.rerank.base import Query, Text
from pygaggle.rerank.transformer import T5Reranker
from transformers import T5Tokenizer

from pathlib import Path
import os
#from pyserini.search import SimpleSearcher
from pyserini.search import pysearch

def main():

    # Build pyserini search
    searcher = pysearch.SimpleSearcher('indexed_clef_pt/')
    
    # Model paths
    t5_model_type = 't5-base'
    t5_model_dir  = 'gs://msmarco-models/1329942/'
    cache_dir     = Path(os.getenv('XDG_CACHE_HOME', str(Path.home() / '.cache'))) / 'covidex'
    flush_cache   = False

    # Model
    device          = torch.device('cuda')
    loader          = CachedT5ModelLoader(t5_model_dir, cache_dir, 'ranker', t5_model_type, flush_cache)
    model           = loader.load().to(device).eval()
    tokenizer       = T5Tokenizer.from_pretrained('unicamp-dl/ptt5-base-portuguese-vocab')
    batch_tokenizer = T5BatchTokenizer(tokenizer, 64, max_length=512)
    reranker        = T5Reranker(model, batch_tokenizer)
    
    # Spacy Senticizer
    nlp = spacy.blank("pt")
    nlp.add_pipe(nlp.create_pipe("sentencizer"))
    
    # Read queries file
    query_file = 'CHAVE/questions_2005_title_desc.txt'
    query_dict = {}
    with open(query_file, mode='r') as reader:
        for line in reader:
            line = line.split('\t')
            query_dict[line[0]] = line[1]


    # Read files for each query
    list_of_files = glob.glob('query_docs/title_desc/*')
    for file in tqdm(list_of_files):
        # query_id is the same for all lines in file
        query_id = file[-3:]
        documents = []
        searcher_hits = []
        query = query_dict[query_id]
                
        with open(file, mode='r') as reader:
            for line in reader:
                doc_id = line.split(' ')[2]
                searcher_hits.append(line)
                document = searcher.doc(doc_id)
                documents.append(document.contents())

        # Creates sliding window
        max_len     = 10 
        stride      = 5   
        text_window = []
        id_window   = []
        for doc_id, doc in enumerate(documents):
            text = nlp(doc)
            sentences = [sent.string.strip() for sent in text.sents]
            for i in range(0, len(sentences), stride):
                segment = ' '.join(sentences[i:i + max_len])
                text_window.append(segment)
                id_window.append(doc_id)

            
        # Reranker using pygaggle
        ranked_results = reranker.rerank(Query(query), [Text(t) for t in text_window])
        
        # Get scores from the reranker
        t5_scores = [r.score for r in ranked_results]
        
        previous        = 0
        max_score       = -10000
        final_t5_scores = []
        total           = 0
        for doc_id, score in zip(id_window, t5_scores):
            if doc_id != previous:
                final_t5_scores.append(max_score)
                previous = doc_id
                max_score = score
                total = 1
            else:
                if score > max_score:
                    max_score = score

        final_t5_scores.append(max_score)
        results = list(zip(searcher_hits, final_t5_scores))
        
        # Sort reranked results by rank
        results.sort(key=lambda x: x[1], reverse=True)
        final_t5_scores.sort(reverse=True)

        # Writes a run file in the TREC format
        output_file = 'results/max/title_desc/query_id_{}'.format(query_id) 
        for i, item in enumerate(results):
            item = str(item[0]).split(' ')
            with open(output_file, mode='a') as writer:
                writer.write(str(" ".join(item[:3])) + ' ' + str(i+1) + ' ' + str(final_t5_scores[i]) + ' Anserini\n')


if __name__ == '__main__':
    main()