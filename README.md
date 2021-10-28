# ia376i_projeto_final

## Referências:

1) Document Ranking with a Pretrained Sequence-to-Sequence Model
https://aclanthology.org/2020.findings-emnlp.63.pdf

3) Neural Pointwise Ranking Baselines on Robust04 - with TPU
https://github.com/castorini/pygaggle/blob/master/docs/experiments-robust04-monot5-tpu.md

3) Anserini: Regressions for Disks 4 & 5 (Robust04)
https://github.com/castorini/anserini/blob/master/docs/regressions-robust04.md

## 1 - Configuração da VM na Google Cloud Platform:

1) Atualização da NVIDIA:
```
wget https://us.download.nvidia.com/tesla/460.73.01/NVIDIA-Linux-x86_64-460.73.01.run
sudo sh NVIDIA-Linux-x86_64-460.73.01.run
```
2) Instalação das ferramentas necessárias:
```
sudo apt-get update
sudo apt-get install git gcc screen --yes
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ./Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
```
3) Criação do ambiente virtual:
```
conda init
conda create --y --name venv36 python=3.6
conda activate venv36
```
4) Instalação do pacote Anserini:
No seu diretório de projeto:
```
sudo apt-get install maven -qq
git clone --recurse-submodules https://github.com/castorini/anserini.git
cd anserini
cd tools/eval && tar xvfz trec_eval.9.0.4.tar.gz && cd trec_eval.9.0.4 && make && cd ../../..
mvn clean package appassembler:assemble -DskipTests -Dmaven.javadoc.skip=true
```
Se a instalação deu certo, você deverá ver anserini-X.Y.Z-SNAPSHOT-fatjar.jar na pasta target:
```
ls target
```
5) Instalação do pacote Pygaggle:
No seu diretório de projeto:
```
git clone --recursive https://github.com/castorini/pygaggle.git
cd pygaggle
pip install .
cd tools/eval && tar xvfz trec_eval.9.0.4.tar.gz && cd trec_eval.9.0.4 && make && cd ../../..
```
6) Garantir que o Java está instalado:
```
sudo apt-get update
sudo apt install default-jdk
```
Com os comandos abaixo, você deverá ter retorno da versão do java e do javac:
```
java -version
javac -version
```
7) Download do dataset Robust04:
```
export DATA_DIR=data/robust04
wget https://raw.githubusercontent.com/castorini/anserini/master/src/main/resources/topics-and-qrels/topics.robust04.txt
wget https://raw.githubusercontent.com/castorini/anserini/master/src/main/resources/topics-and-qrels/qrels.robust04.txt
wget https://storage.googleapis.com/castorini/robust04/trec_disks_4_and_5_concat.txt
cd ../../
```
## 2 - Pipeline:
### 2.1 - Indexação
As opções de indexação podem ser encontradas no link: https://github.com/castorini/anserini/blob/master/docs/common-indexing-options.md

Nesse projeto, ela foi realizada com o código abaixo:
```
export DATA_DIR=/home/l224016/projects/ia376_projeto_final/data/robust04
sh target/appassembler/bin/IndexCollection -collection TrecCollection \
 -input ${DATA_DIR} \
 -index indexes/lucene-index.robust04.pos+docvectors+raw+unique \
 -generator DefaultLuceneDocumentGenerator \
 -threads 16 -storePositions -storeDocvectors -storeRaw -uniqueDocid
```
A opção -uniqueDocId foi utilizada, pois parece que o documento FT911-1 aparece em duplicidade no corpus.

### 2.2 - Retrieval do BM25:
São utilizados como input os arquivos contendo as queries topics.robust04.txt e a pasta de indexação, presente: em indexes/lucene-index.robust04.pos+docvectors+raw+unique.

Para gerar os arquivos run.robust04.bm25.topics.robust04.XXXXhits.txt para todas as quantidades de textos candidatos {1k, 2k, 3k, 4k, 5k}, é necessário rodar o comando abaixo alterando o parâmetro -hits e ir gerando os outputs correspondentes.

Abaixo, é gerado um arquivo de saída run com 1000 hits (textos candidatos) para cada uma das queries.
```
export DATA_DIR=/home/l224016/projects/ia376_projeto_final/data/robust04
cd anserini
target/appassembler/bin/SearchCollection -index indexes/lucene-index.robust04.pos+docvectors+raw+unique \
-topicreader Trec -topics ${DATA_DIR}/topics.robust04.txt \
-hits 1000 \
-output ${DATA_DIR}/runs/run.robust04.bm25.topics.robust04.1000hits.txt \
-bm25
```
### 2.3 - Avaliação dos resultados do BM25:
É utilizada a ferramenta trec_eval que está presente tanto dentro do diretorio tools/eval/trec_eval.9.0.4/ da pasta anserini quanto da pasta pygaggle.

A etapa de avaliação necessita das queries anotadas com documentos relevantes à elas presentes no arquivo qrels.robust04.txt.

Abaixo, um exemplo de avaliação do arquivo de saída run com 1000hits:
```
export DATA_DIR=/home/l224016/projects/ia376_projeto_final/data/robust04
cd anserini
tools/eval/trec_eval.9.0.4/trec_eval -m map -m ndcg_cut.20 ${DATA_DIR}/qrels.robust04.txt ${DATA_DIR}/runs/run.robust04.bm25.topics.robust04.1000hits.txt
```

### 2.4 - Criação dos arquivos query_doc_ids_XXXXhits.tsv e segment_texts_XXXXhits.txt:
Essa etapa consiste na segmentação dos textos candidatos em passagens para que possam servir de input ao modelo monoT5-base, dado que o modelo aceita como entrada até 512 tokens sem realizar o truncamento.

Para realizar essa segmentação e criação dos arquivos, é necessário o código: create_robust04_monot5_input.py. Será necessário criar tais paredes arquivos para cada quantidade de hits que desejar avaliar.

É possível executá-lo com a sua venv ativada, conforme abaixo:
```
cd codigos
python create_robust04_monot5_input.py
```

## 2.5 Execução do monoT5-base a partir dos arquivos de segmentação:
Agora que já temos os arquivos segmentados, podemos utilizar o monoT5-base do Pygaggle para reranquear os documentos candidatos. O código utilizado a seguir já faz o reranqueamento para cada passagem de cada texto e assume o maior score de todos os segmentos de um texto como sendo o score do texto. Feito isso, o código gera os resultados já no formato do Trec Eval. Será necessário fazer o reranqueamento para cada quantidade de hits que desejar avaliar.

É possível executá-lo com a sua venv ativada, conforme abaixo:
```
cd codigos
python t5_reranker_robust04.py
```

## 2.6 Avaliação dos resultados do monoT5-base:
Com o reranqueamento gerado pelo código t5_reranker_robust04.py já no formato do Trec Eval é possível então avaliar os resultados com relação às métricas MAP e nDCG20. Será necessário fazer a avaliação para cada reranqueamento com quantidade de hits diferentes.

Um exemplo de avaliação do reranquemaneto com -hits = 1000 pode ser realizado com o código abaixo.
```
export DATA_DIR=/home/l224016/projects/ia376_projeto_final/data/robust04
cd pygaggle
tools/eval/trec_eval.9.0.4/trec_eval -m map -m ndcg_cut.20 ${DATA_DIR}/qrels.robust04.txt ${DATA_DIR}/monot5/monot5_results_1000hits.txt
```
