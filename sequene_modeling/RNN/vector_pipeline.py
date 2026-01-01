import torch
import torch.nn as nn 
import numpy as np
from collections import Counter

class Vocabulary:
    def __init__(self):
        self.w2id = {}
        self.id2w = {}
        self.word_count = Counter()
        
        
        #Special tokens
        self.PAD_TOKEN = '<PAD>'
        self.UNK_TOKEN = '<UNK>'
        # self.SOS_TOKEN = '<SOS>'
        # self.EOS_TOKEN = '<EOS>'
        
        self.PAD_ID = 0
        self.UNK_ID = 1
        # self.SOS_ID = 2
        # self.EOS_ID = 3
    
    def build_vocab(self, sentences, min_freq = 1):
        
        #count word frequencies
        for sentence in sentences:
            self.word_count.update(sentence)
        
        # Add special tokens
        self.w2id[self.PAD_TOKEN] = self.PAD_ID
        self.w2id[self.UNK_TOKEN] = self.UNK_ID
        # self.w2id[self.SOS_TOKEN] = self.SOS_ID
        # self.w2id[self.EOS_TOKEN] = self.EOS_ID
        
        #add words that meet min frequency
        idx = 3
        for word, freq in self.word_count.most_common():
            if freq >= min_freq:
                self.w2id[word] = idx
                idx += 1
        #create reverse mapping
        self.id2w = {idx : word for word, idx in self.w2id.items()}
        
        print(f"Vocabulary built with {len(self.w2id)} tokens.")
        
    
    


if __name__ == "__main__":
    sentences = [
        ["hello", "world"],
        ["hello", "there"],
        ["goodbye", "world"]
    ]
    
    vocab = Vocabulary()
    vocab.build_vocab(sentences, min_freq=1)
    
    print(vocab.w2id)
    print(vocab.id2w)