from collections import Counter
from multiprocessing import Pool, cpu_count
from pathlib import Path
from pyfaidx import Fasta
import numpy as np
from tqdm import tqdm
from tokenizers import models, Tokenizer, decoders

# Load the fasta file
fasta_file = Path("/scratch/kyle/hyena-dna/data/hg38/hg38.ml.fa")
seqs = Fasta(str(fasta_file))

# Vocabulary mapping
vocab = {"N": 0, "A": 1, "C": 2, "G": 3, "T": 4}

# Convert sequences to list of ids
seqs = list("".join(str(seqs[chr_name][:]) for chr_name in seqs.keys()))
ids = np.array([vocab.get(c, 0) for c in seqs])

# Store the original length of ids for compression rate calculation
original_length = len(ids)

# Function to get pair statistics with boundary handling
def get_stats(ids_chunk, prev_last_id=None):
    counts = Counter(zip(ids_chunk[:-1], ids_chunk[1:]))
    if prev_last_id is not None:
        counts[(prev_last_id, ids_chunk[0])] += 1
    return counts

# Function to merge pairs in ids using numpy
def merge(ids, pair, idx):
    new_ids = []
    skip = False
    for i in range(len(ids) - 1):
        if skip:
            skip = False
            continue
        if ids[i] == pair[0] and ids[i + 1] == pair[1]:
            new_ids.append(idx)
            skip = True
        else:
            new_ids.append(ids[i])
    if not skip:
        new_ids.append(ids[-1])
    return np.array(new_ids)

# BPE merging with parallel processing
def bpe_parallel(ids, vocab_size):
    merges = {}
    num_cores = cpu_count()
    chunk_size = len(ids) // num_cores

    pbar = tqdm(range(len(vocab), vocab_size))
    for i in pbar:
        # Split ids into chunks for parallel processing
        chunks = [ids[j:j + chunk_size] for j in range(0, len(ids), chunk_size)]
        last_ids = [chunk[-1] for chunk in chunks[:-1]]  # Last ids for all chunks except the last

        # Get pair statistics in parallel
        with Pool() as pool:
            stats_list = pool.starmap(get_stats, [(chunks[k], last_ids[k-1] if k > 0 else None) for k in range(len(chunks))])
        stats = sum(stats_list, Counter())

        # Find the most frequent pair
        pair = max(stats, key=stats.get)

        # Merge the most frequent pair
        ids = merge(ids, pair, i)
        merges[pair] = i

        # Calculate compression rate
        compression_rate = len(ids) / original_length
        pbar.set_postfix({"Compression Rate": f"{compression_rate:.2%}"})

    return ids, merges

# Execute BPE
ids, merges = bpe_parallel(ids, vocab_size=4096)

# Create reverse vocabulary
r_vocab = {v: k for k, v in vocab.items()}
for (p0, p1), idx in merges.items():
    r_vocab[idx] = r_vocab[p0] + r_vocab[p1]
vocab = {v: k for k, v in r_vocab.items()}

# Create merge list
merges = [(r_vocab[p0], r_vocab[p1]) for (p0, p1), _ in merges.items()]

# Save the tokenizer
tokenizer = Tokenizer(models.BPE(vocab=vocab, merges=merges))
tokenizer.decoder = decoders.ByteLevel()
tokenizer.save("tokenizer.json")