import os
import json
import numpy
import re
import torch
import pdb


class Vocabulary:
    def __init__(self):
        self.max_size = 100
        self.vocab = {}

    def __getitem__(self, token):
        if not (token in self.vocab.keys()):
            if len(self.vocab) >= self.max_size:
                raise ValueError("Maximum vocabulary capacity reached")
            self.vocab[token] = len(self.vocab) + 1
        return self.vocab[token]

    def copy_vocab_from(self, other):
        '''
        Copy the vocabulary of another Vocabulary object to the current object.
        '''
        self.vocab.update(other.vocab)


class InstructionsPreprocessor(object):
    def __init__(self):
        self.vocab = Vocabulary()

    def __call__(self, mission):
        max_instr_len = 0
        tokens = re.findall("([a-z]+)", mission.lower())
        return numpy.array([self.vocab[token] for token in tokens])
        #instr)
        '''
        max_instr_len = max(len(instr), max_instr_len)

        instrs = numpy.zeros((1, max_instr_len))

        for i, instr in enumerate(raw_instrs):
            instrs[i, :len(instr)] = instr

        instrs = torch.tensor(instrs, dtype=torch.long).squeeze()
        '''
        return raw_instrs