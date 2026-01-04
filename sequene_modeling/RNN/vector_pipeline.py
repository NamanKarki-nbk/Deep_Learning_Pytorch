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
        self.SOS_TOKEN = '<SOS>'
        self.EOS_TOKEN = '<EOS>'
        
        self.PAD_ID = 0
        self.UNK_ID = 1
        self.SOS_ID = 2
        self.EOS_ID = 3

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
        idx = 4
        for word, freq in self.word_count.most_common():
            if freq >= min_freq:
                self.w2id[word] = idx
                idx += 1
        #create reverse mapping
        self.id2w = {idx : word for word, idx in self.w2id.items()}
        
        print(f"Vocabulary built with {len(self.w2id)} tokens.")
        
    def encode(self, sentence, add_sos = False, add_eos = False):
        
        ids = []
        
        if add_sos:
            ids.append(self.SOS_ID)
        
        for word in sentence:
            ids.append(self.w2id.get(word,self.UNK_ID))

        if add_eos:
            ids.append(self.EOS_ID)

        return ids

    def decode(self, ids, remove_special = True):
        
        words = [self.id2w.get(idx, self.UNK_TOKEN) for idx in ids]
        
        if remove_special:
            specials = {self.PAD_TOKEN, self.UNK_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN}
            words = [word for word in words if word not in specials]
        
        return words
    
    def __len__(self):
        return len(self.w2id)
    
def pad_sequences(sequences, pad_id = 0, max_len = None):
    
    if max_len is None:
        max_len = max(len(seq) for seq in sequences)
    
    padded = []
    for seq in sequences:
        if len(seq) < max_len:
            seq = seq + [pad_id] * (max_len - len(seq))
        else:
            seq = seq[:max_len]
        padded.append(seq)
    
    return torch.LongTensor(padded)

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
    ids=[]
    for sentence in sentences:
        ids.append(vocab.encode(sentence))
        print(ids)
    
    for id in ids:
        words = vocab.decode(id)
        print(words)
    