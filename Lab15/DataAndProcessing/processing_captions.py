import pandas as pd
import numpy as np
import pickle
import re
import os

def load_glove_embeddings(path):
    df = pd.read_csv(
        path,
        sep=r"\s+|\t|,",
        engine="python",
        header=None,
        on_bad_lines="skip"
    )

    embeddings_dict = {}
    for row in df.itertuples(index=False):
        word = str(row[0])
        vector = np.array(row[1:], dtype="float32")
        embeddings_dict[word] = vector
    print(f"Loaded {len(embeddings_dict)} word vectors.")
    return embeddings_dict


def clean_and_tokenize(caption):
    caption = caption.lower()
    caption = re.sub(r"[^a-z0-9\s]", "", caption)
    tokens = caption.strip().split()
    return tokens


def main():
    embeddings_dict = load_glove_embeddings(
        "wiki_giga_2024_50_MFT20_vectors_seed_123_alpha_0.75_eta_0.075_combined.txt"
    )

    df = pd.read_csv("../../data/Flick8K/captions.txt", sep=",")
    print(f"Loaded {len(df)} captions.")

    all_tokens = []
    for caption in df["caption"]:
        tokens = clean_and_tokenize(str(caption))
        all_tokens.extend(tokens)

    unique_words = sorted(set(all_tokens))
    print(f"Unique words in captions: {len(unique_words)}")

    # Add special tokens
    special_tokens = ["<PAD>", "<START>", "<END>", "<UNK>"]
    vocab = {tok: idx for idx, tok in enumerate(special_tokens)}
    inv_vocab = {idx: tok for tok, idx in vocab.items()}

    # Add unique words to vocab
    for word in unique_words:
        if word not in vocab:
            idx = len(vocab)
            vocab[word] = idx
            inv_vocab[idx] = word

    embed_size = len(next(iter(embeddings_dict.values())))
    vocab_size = len(vocab)
    embedding_matrix = np.zeros((vocab_size, embed_size), dtype="float32")

    for word, idx in vocab.items():
        if word in embeddings_dict:
            embedding_matrix[idx] = embeddings_dict[word]
        else:
            embedding_matrix[idx] = np.random.normal(scale=0.6, size=(embed_size,))

    np.save("embedding_matrix.npy", embedding_matrix)
    print(f"Saved embedding_matrix.npy of shape {embedding_matrix.shape}")

    ground_truth = []
    for _, row in df.iterrows():
        image_name = row["image"].strip()
        caption_text = str(row["caption"]).strip()
        tokens = clean_and_tokenize(caption_text)

        token_ids = [vocab["<START>"]]
        for tok in tokens:
            token_ids.append(vocab.get(tok, vocab["<UNK>"]))
        token_ids.append(vocab["<END>"])

        ground_truth.append({
            "image": image_name,
            "caption_tokens": token_ids
        })

    ground_truth_df = pd.DataFrame(ground_truth)
    ground_truth_df.to_pickle("ground_truth.pkl")
    print(f"Saved {len(ground_truth_df)} processed captions to ground_truth.pkl")

    with open("vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    with open("inv_vocab.pkl", "wb") as f:
        pickle.dump(inv_vocab, f)

    print(f"Saved vocab.pkl and inv_vocab.pkl ({len(vocab)} words).")
    print("\nAll files saved to current directory!")


if __name__ == "__main__":
    main()
