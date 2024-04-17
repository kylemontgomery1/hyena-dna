from pathlib import Path
from pyfaidx import Fasta
import polars as pl
import pandas as pd
import torch
from random import randrange, random
import numpy as np
import os

from src.dataloaders.datasets.hg38_char_tokenizer import CharacterTokenizer
from src.dataloaders.datasets.hg38_dataset import FastaInterval


class GeneIdentificationDataset(torch.utils.data.Dataset):
    
    def __init__(
        self,
        split,
        bed_file,
        fasta_file,
        ref_labels_file,
        max_length,
        pad_max_length=None,
        tokenizer=None,
        tokenizer_name=None,
        add_eos=False,
        return_seq_indices=False,
        shift_augs=None,
        rc_aug=False,
        return_augs=False,
        replace_N_token=False,  # replace N token with pad token
        pad_interval = False,  # options for different padding
        d_output=None,
    ):
        
        self.max_length = max_length
        self.pad_max_length = pad_max_length if pad_max_length is not None else max_length
        self.tokenizer_name = tokenizer_name
        self.tokenizer = tokenizer
        self.return_augs = return_augs
        self.add_eos = add_eos
        self.replace_N_token = replace_N_token  
        self.pad_interval = pad_interval
        self.d_output = d_output         
        
        bed_path = Path(bed_file)
        assert bed_path.exists(), 'path to .bed file must exist'

        # read bed file
        df_raw = pd.read_csv(str(bed_path), sep = '\t', names=['chr_name', 'start', 'end', 'split'])
        # select only split df
        self.df = df_raw[df_raw['split'] == split]

        self.fasta = FastaInterval(
            fasta_file = fasta_file,
            # max_length = max_length,
            return_seq_indices = return_seq_indices,
            shift_augs = shift_augs,
            rc_aug = rc_aug,
            pad_interval = pad_interval,
        )
        
        ref_label_path = Path(ref_labels_file)
        assert ref_label_path.exists(), 'path to reference labels file must exist'
        df_raw = pd.read_csv(str(ref_label_path), sep='\t', comment='#', header=None, 
                 names=['seqname', 'source', 'feature', 'start', 'end', 'score', 'strand', 'frame', 'attribute'])
        df_raw['start'] = df_raw['start'].astype(int)
        df_raw['end'] = df_raw['end'].astype(int)
        df_raw['end'] += 1
        self.labels = df_raw[['seqname', 'start', 'end']]


    def __len__(self):
        return len(self.df)

    def replace_value(self, x, old_value, new_value):
        return torch.where(x == old_value, new_value, x)

    def __getitem__(self, idx):
        """Returns a sequence of specified len"""
        # sample a random row from df
        row = self.df.iloc[idx]
        # row = (chr, start, end, split)
        chr_name, start, end = (row[0], row[1], row[2])
        print(chr_name)
        print(start)
        print(end)
        seq = self.fasta(chr_name, start, end, max_length=self.max_length, return_augs=self.return_augs)
        print(len(seq))
        if self.tokenizer_name == 'char':

            seq = self.tokenizer(seq,
                add_special_tokens=True if self.add_eos else False,  # this is what controls adding eos
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
            )
            seq = seq["input_ids"]  # get input_ids

        elif self.tokenizer_name == 'bpe':
            seq = self.tokenizer(seq, 
                # add_special_tokens=False, 
                padding="max_length",
                max_length=self.pad_max_length,
                truncation=True,
            ) 
            # get input_ids
            if self.add_eos:
                seq = seq["input_ids"][1:]  # remove the bos, keep the eos token
            else:
                seq = seq["input_ids"][1:-1]  # remove both special tokens
        
        # convert to tensor
        seq = torch.LongTensor(seq)  # hack, remove the initial cls tokens for now
        print(seq)
        print(seq.shape)
        if self.replace_N_token:
            # replace N token with a pad token, so we can ignore it in the loss
            seq = self.replace_value(seq, self.tokenizer._vocab_str_to_int['N'], self.tokenizer.pad_token_id)

        targets = torch.zeros_like(seq)
        start, end = int(start), int(end)
        
        t = self.labels[
            (self.labels['seqname'] == chr_name) & 
            (
                ((self.labels['start'] >= start) & (self.labels['end'] < end)) | # gene is within the interval
                ((self.labels['start'] < start) & (self.labels['end'] >= start)) | # gene starts before the interval
                ((self.labels['start'] < end) & (self.labels['end'] >= end)) # gene ends after the interval
            )
        ]
        for rows in t.itertuples():
            print(rows)
            start_idx = max(start, rows.start) - start + 1
            end_idx = min(end, rows.end) - start + 1
            targets[start_idx:end_idx] = 1 

        return seq.clone(), targets.clone()
    
if __name__ == '__main__':
    
    tokenizer = CharacterTokenizer(
        characters=['A', 'C', 'G', 'T', 'N'],
        model_max_length=1000002,
        add_special_tokens=False,
        padding_side='right',
    )
    
    dataset = GeneIdentificationDataset(
        split='train',
        bed_file='./data/hg38/human-sequences.bed',
        fasta_file='./data/hg38/hg38.ml.fz',
        ref_labels_file='./data/hg38/hg38.refGene.gtf',
        max_length=1000000,
        pad_max_length=1000000,
        tokenizer=tokenizer,
        tokenizer_name='char',
        add_eos=True,
        return_seq_indices=False,
        shift_augs=None,
        rc_aug=False,
        return_augs=False,
        replace_N_token=False, 
        pad_interval = False,  
    )
    
    print(dataset[0])
        
    
    
    
        
        
        
