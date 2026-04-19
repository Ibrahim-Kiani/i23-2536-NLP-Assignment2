import subprocess


def run(cmd):
    print("\n>>", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    run(["python", "src/part1_tfidf_ppmi.py", "--cleaned", "cleaned.txt", "--metadata", "Metadata.json", "--out_dir", "embeddings"])
    run(["python", "src/part1_skipgram.py", "--corpus", "cleaned.txt", "--tag", "cleaned", "--dim", "100", "--epochs", "5"])
    run(["python", "src/part1_eval.py", "--cleaned", "cleaned.txt", "--raw", "raw.txt", "--out_dir", "embeddings"])

    run(["python", "src/part2_prepare.py", "--cleaned", "cleaned.txt", "--metadata", "Metadata.json", "--out_dir", "data"])
    run(["python", "src/part2_bilstm.py", "--task", "pos", "--word2idx", "embeddings/word2idx.json", "--emb", "embeddings/embeddings_w2v.npy"])
    run(["python", "src/part2_bilstm.py", "--task", "ner", "--word2idx", "embeddings/word2idx.json", "--emb", "embeddings/embeddings_w2v.npy"])

    run(["python", "src/part3_prepare.py", "--cleaned", "cleaned.txt", "--metadata", "Metadata.json", "--word2idx", "embeddings/word2idx.json", "--out", "data/topic_cls.npz"])
    run(["python", "src/part3_transformer.py", "--data", "data/topic_cls.npz", "--word2idx", "embeddings/word2idx.json"])
