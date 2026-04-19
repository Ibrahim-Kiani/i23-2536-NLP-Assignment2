# i23-XXXX-NLP-Assignment2

PyTorch-from-scratch implementation for:
- Part 1: TF-IDF, PPMI, Skip-gram Word2Vec
- Part 2: POS + NER sequence labeling with 2-layer BiLSTM (custom CRF for NER)
- Part 3: Transformer encoder topic classification (custom attention and encoder stack)

## Setup

```bash
pip install -r requirements.txt
```

## Run Part 1

```bash
python src/part1_tfidf_ppmi.py --cleaned cleaned.txt --metadata Metadata.json --out_dir embeddings
python src/part1_skipgram.py --corpus cleaned.txt --tag cleaned --dim 100 --epochs 5
python src/part1_eval.py --cleaned cleaned.txt --raw raw.txt --out_dir embeddings
```

Outputs:
- embeddings/tfidf_matrix.npy
- embeddings/ppmi_matrix.npy
- embeddings/embeddings_w2v.npy
- embeddings/word2idx.json

## Run Part 2

```bash
python src/part2_prepare.py --cleaned cleaned.txt --metadata Metadata.json --out_dir data
python src/part2_bilstm.py --task pos --word2idx embeddings/word2idx.json --emb embeddings/embeddings_w2v.npy
python src/part2_bilstm.py --task ner --word2idx embeddings/word2idx.json --emb embeddings/embeddings_w2v.npy
```

Outputs:
- data/pos_*.conll
- data/ner_*.conll
- models/bilstm_pos_*.pt
- models/bilstm_ner_*.pt

## Run Part 3

```bash
python src/part3_prepare.py --cleaned cleaned.txt --metadata Metadata.json --word2idx embeddings/word2idx.json --out data/topic_cls.npz
python src/part3_transformer.py --data data/topic_cls.npz --word2idx embeddings/word2idx.json
```

Outputs:
- models/transformer_cls.pt
- models/transformer_metrics.json
- models/transformer_curves.png

## Notebook

Notebook entry point:
- notebooks/i23-XXXX_Assignment2_DS-X.ipynb

## Repository URL

Replace this with your final public URL after push:
- https://github.com/<your-username>/i23-XXXX-NLP-Assignment2
