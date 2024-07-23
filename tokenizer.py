from pathlib import Path
from pyfaidx import Fasta
import pandas as pd
import torch
from random import randrange, random
import numpy as np
import os
from transformers import AutoTokenizer
from tqdm import tqdm
import matplotlib.pyplot as plt
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)

fasta_file = Path("/scratch/kyle/hyena-dna/data/hg38/hg38.ml.fa")
seqs = Fasta(str(fasta_file))

vocab = {
    "N": 0,
    "A": 1,
    "C": 2,
    "G": 3,
    "T": 4,
}

seqs = list("".join(str(seqs[chr_name][:]) for chr_name in seqs.keys()))
ids = [vocab.get(c, 0) for c in seqs]

def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        if 0 not in pair: # refusing to merge N tokens
            counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
    newids = []
    i = 0
    while i < len(ids):
        if i<len(ids)-1 and ids[i]==pair[0] and ids[i+1]==pair[1]:
            newids.append(idx)
            i+=2
        else:
            newids.append(ids[i])
            i+=1
    return newids

vocab_size = 4096
merges = {} # List[Tuple[int, int], int]

for i in tqdm(range(len(vocab), vocab_size)):
    stats = get_stats(ids)
    pair = max(stats, key=stats.get)
    ids = merge(ids, pair, i)
    merges[pair] = i
    
r_vocab = {v:k for k,v in vocab.items()}

for (p0, p1), idx in merges.items():
    r_vocab[idx] = r_vocab[p0] + r_vocab[p1]
    
vocab = {v:k for k,v in r_vocab.items()}

merges = [(r_vocab[p0], r_vocab[p1]) for (p0, p1), _ in merges.items()]

tokenizer = Tokenizer(models.BPE(vocab=vocab, merges=merges))

tokenizer.decoder = decoders.ByteLevel()

tokenizer.save("tokenizer.json")