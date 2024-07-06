from pathlib import Path
from pyfaidx import Fasta
import polars as pl
import pandas as pd
import torch
from random import randrange, random
import numpy as np
import os

class FastaInterval():
    def __init__(
        self,
        fasta_file,
    ):
        fasta_file = Path(fasta_file)
        assert fasta_file.exists(), 'path to fasta file must exist'

        self.seqs = Fasta(str(fasta_file))

        # calc len of each chromosome in fasta file, store in dict
        self.chr_lens = {}
        for chr_name in self.seqs.keys():
            self.chr_lens[chr_name] = len(self.seqs[chr_name])


    def __call__(self, chr_name, start, end, max_length, fill_to_max_length, fill_side):
        """
        max_length passed from dataset, not from init
        """
        interval_length = end - start
        chromosome = self.seqs[chr_name]
        chromosome_length = self.chr_lens[chr_name]
        
        if fill_to_max_length and interval_length < max_length:
            if fill_side == 'left':
                start = max(0, start - (max_length - interval_length))
            elif fill_side == 'right':
                end = min(chromosome_length, end + (max_length - interval_length))
            else:
                raise ValueError(f"fill_side must be either 'left' or 'right', got {fill_side}")
        
        seq = str(chromosome[start:end])
        return seq, start, end
                
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
        d_output,
        pad_to_max_length,
        truncate_to_max_length,
        fill_side,
        fill_to_max_length,
        length_multiplier,
    ):
        self.tokenizer = tokenizer
        self.tokenizer_name = tokenizer_name
        self.max_length = max_length
        self.d_output = d_output
        self.pad_to_max_length = pad_to_max_length
        self.truncate_to_max_length = truncate_to_max_length
        self.fill_side = fill_side
        self.fill_to_max_length = fill_to_max_length
        self.length_multiplier = length_multiplier
        
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
        
        seq, start, end = self.fasta(chr_name, start, end, max_length=self.length_multiplier*self.max_length, fill_to_max_length=self.fill_to_max_length, fill_side=self.fill_side)
        
        seq_tokenized = self.tokenizer(seq,
            add_special_tokens=False, 
            max_length=self.max_length,
            padding="max_length" if self.pad_to_max_length else False,
            truncation=self.truncate_to_max_length,
            return_offsets_mapping=self.tokenizer_name == 'bpe',
        )
        seq = torch.LongTensor(seq_tokenized["input_ids"]) 
        
        print(f"{len(seq)=}")
        
        # computing padding offset
        offset = (seq == self.tokenizer.pad_token_id).sum().item()
        
        # if no padding, but truncation was done on the left, we need to adjust the start position
        if offset == 0 and self.tokenizer.truncation_side == 'left':
            start = end - len(self.tokenizer.batch_decode(seq, skip_special_tokens=True)[0])
        
        targets = torch.zeros_like(seq) if self.tokenizer_name == 'char' else torch.zeros(offset + end - start, dtype=torch.long)
        
        t = self.labels[
            (self.labels['seqname'] == chr_name) & 
            (
                ((self.labels['start'] >= start) & (self.labels['end'] < end)) | # gene is within the interval
                ((self.labels['start'] < start) & (self.labels['end'] >= start)) | # gene starts before the interval
                ((self.labels['start'] < end) & (self.labels['end'] >= end)) # gene ends after the interval
            )
        ]
        
        for row in t.itertuples():
            start_idx = max(row.start - start + offset, offset)
            end_idx = min(start_idx + (row.end - row.start), seq.size(0))
            targets[start_idx:end_idx] = 1
            
        if self.tokenizer_name == 'bpe': # combining targets for BPE tokenization
            new_targets = torch.zeros_like(seq)
            for tok_idx, (tok_start, tok_end) in enumerate(seq_tokenized['offset_mapping']):
                if tok_start == 0 and tok_end == 0: # pad token
                    new_targets[tok_idx] = -100
                else:
                    new_targets[tok_idx] = 1 if targets[offset+tok_start:offset+tok_end].sum() > 0 else 0 
            targets = new_targets

        return seq.clone(), targets.clone()
