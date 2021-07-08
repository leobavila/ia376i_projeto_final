import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Process a bm25 returning amount checker.')
parser.add_argument('--hits', type=str, help='Amount of candidate  texts that bm25 will return')

args = parser.parse_args()

if __name__ == '__main__':
    filename ="~/projects/ia376_projeto_final/anserini/runs/run.robust04.bm25.topics.robust04.{}hits.txt".format(args.hits)
    bm25 = pd.read_csv(filename, sep = " ", names = ['id', 'Q0', 'docid', 'rank', 'score', 'model'])
    hits = bm25["rank"].max()
    print("Reading file:\n{}".format(filename.split("/")[-1]))
    print("Number of candidate texts: {}".format(hits))
