from pathlib import Path
from pyfaidx import Fasta
import polars as pl
import pandas as pd
import torch
from random import randrange, random
import numpy as np
import os

class FastaInterval():
    
    def __init__(self, fasta_file):
        fasta_file = Path(fasta_file)
        assert fasta_file.exists(), 'path to fasta file must exist'
        self.seqs = Fasta(str(fasta_file))

    def __call__(self, chr_name, start, end):
        """
        max_length passed from dataset, not from init
        """        
        seq = str(self.seqs[chr_name][start:end])
        return seq
                
class GeneIdentificationDataset(torch.utils.data.Dataset):
    
    def __init__(
        self,
        split,
        bed_file,
        fasta_file,
        ref_labels_file,
        tokenizer,
        tokenizer_name,
        max_length,
    ):
        self.tokenizer = tokenizer
        self.tokenizer_name = tokenizer_name
        self.max_length = max_length
        
        bed_path = Path(bed_file)
        assert bed_path.exists(), 'path to .bed file must exist'
        df_raw = pd.read_csv(str(bed_path), sep = '\t', names=['chr_name', 'start', 'end', 'split'])
        self.df = df_raw[df_raw['split'] == split]

        self.fasta = FastaInterval(
            fasta_file = fasta_file,
        )
        
        ref_label_path = Path(ref_labels_file)
        assert ref_label_path.exists(), 'path to reference labels file must exist'
        df_raw = pd.read_csv(str(ref_label_path), sep='\t', comment='#', header=None, 
                 names=['seqname', 'source', 'feature', 'start', 'end', 'score', 'strand', 'frame', 'attribute'])
        df_raw = df_raw[df_raw['feature'] == 'transcript']
        df_raw['start'] = df_raw['start'].astype(int)
        df_raw['start'] -= 1
        df_raw['end'] = df_raw['end'].astype(int)
        self.labels = df_raw[['seqname', 'start', 'end']]


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """Returns a sequence of specified len"""
        row = self.df.iloc[idx] # row = (chr, start, end, split)
        chr_name, start, end = (row[0], row[1], row[2])
        
        seq = self.fasta(chr_name, start, end)
        
        seq_tokenized = self.tokenizer(seq,
            add_special_tokens=False, 
            max_length=self.max_length,
            padding="max_length", # tokenizer will pad to the right
            truncation=False,
            return_offsets_mapping=True,
        )
        seq = torch.tensor(seq_tokenized["input_ids"]).long()
        
        # filter labels to those genes which intersect with our seq
        t = self.labels[
            (self.labels['seqname'] == chr_name) & 
            (
                ((self.labels['start'] >= start) & (self.labels['end'] < end)) | # gene is within the interval
                ((self.labels['start'] < start) & (self.labels['end'] >= start)) | # gene starts before the interval
                ((self.labels['start'] < end) & (self.labels['end'] >= end)) # gene ends after the interval
            )
        ]
        
        # construct char-level targets
        char_targets = torch.zeros((end-start,)).long() 
        for row in t.itertuples():
            start_idx = max(row.start - start, 0)
            end_idx = min(start_idx + (row.end - row.start), seq.size(0)) # labels could overflow onto pad_tokens, but this is fixed below
            char_targets[start_idx:end_idx] = 1
        
        # aggregate char-level targets to token-level targets
        targets = torch.zeros_like(seq)
        for tok_idx, (tok_start, tok_end) in enumerate(seq_tokenized['offset_mapping']):
            if tok_start == tok_end: # ignore loss on pad token
                targets[tok_idx] = -100
            if tok_end == tok_start + 1 and seq[tok_idx].item() == self.tokenizer.unk_token_id: # ignore loss on unk_token likely "N"
                targets[tok_idx] = -100
            else:
                targets[tok_idx] = 1 if char_targets[tok_start:tok_end].sum() > 0 else 0 

        return seq.clone(), targets.clone()
