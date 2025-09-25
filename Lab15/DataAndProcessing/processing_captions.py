import pandas as pd
import numpy as np


def load_glove_embeddings(path):
    #load with pandas and handle multiple delimiters
    df = pd.read_csv(
        path,
        sep=r"\s+|\t|,",  #split on multiple spaces, tabs, or commas
        engine="python",
        header=None,
        on_bad_lines="skip"
    )

    embeddings_dict = {}
    for row in df.itertuples(index=False):
        word = row[0]
        vector = np.array(row[1:], dtype="float32")
        embeddings_dict[word] = vector

    return embeddings_dict


def main():
    #load embeddings
    embeddings_dict = load_glove_embeddings(
        "wiki_giga_2024_50_MFT20_vectors_seed_123_alpha_0.75_eta_0.075_combined.txt"
    )
    print(f"Loaded {len(embeddings_dict)} word vectors.")

    #load captions
    ground_truth = []
    with open("captions.txt", "r") as f:
        next(f)  # skip header
        for line in f:
            values = line.strip().split(",")
            if len(values) < 2:
                continue
            image_name = values[0].strip()
            caption_text = ",".join(values[1:]).strip()
            words = caption_text.split()

            #get embeddings for words in the caption
            caption_embeddings = []
            for word in words:
                if word in embeddings_dict:
                    caption_embeddings.append(embeddings_dict[word])

            if caption_embeddings:  # only include captions with embeddings
                ground_truth.append({
                    "image": image_name,
                    "caption_embeddings": np.stack(caption_embeddings)  # [seq_len, embed_size]
                })

    print(f"Processed {len(ground_truth)} captions with embeddings.")

    #save to pickle
    df = pd.DataFrame(ground_truth)
    df.to_pickle("ground_truth.pkl")
    print("Ground truth saved to ground_truth.pkl")


if __name__ == "__main__":
    main()
