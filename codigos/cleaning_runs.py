import os
import pandas as pd

input = "/home/l224016/projects/ia376_projeto_final/anserini/runs"
output = "/home/l224016/projects/ia376_projeto_final/anserini/runs/clean_runs"

def main():
    filelist = []
    for file in os.listdir(input):
        if file.endswith(".txt"):
            filelist.append(file)

    for file in filelist:
        input_path = os.path.join(input, file)
        output_path = os.path.join(output, file)

        ranking_list = pd.read_csv(input_path, sep = " ", names = ['id', 'Q0', 'docid', 'rank', 'score', 'model'])
        ranking_list = ranking_list.drop_duplicates(["id", "docid"])
        ranking_list.to_csv(output_path, sep = " ", index = False, header = None)

if __name__ == '__main__':
    main()
