import logging
import re
import os
import shutil
from pathlib import Path
from typing import Union, Dict, List

import random

import flair
from flair.data import Corpus, Dictionary, MultiCorpus, FlairDataset, Sentence, Token
from flair.datasets.base import find_train_dev_test_files
from flair.file_utils import cached_path, unpack_file, unzip_file

log = logging.getLogger("flair")


class ToyCorpus(Corpus):

    def __init__(
            self, 
            vocab_length: int = 100,
            tag_set_size: int = 20,
            max_sentence_length: int = 40,
            total_sentence_count: int = 1000,
            name: str = 'ToyCorpus',
            label: str = 'toy-tag'
    ):
        self.whole_dataset = Toy5Dataset(vocab_length, tag_set_size, max_sentence_length, total_sentence_count, label)
        self.min_freq = 1
        super().__init__(self.whole_dataset, name=name)
    
    def make_tag_dictionary(self, tag_type: str) -> Dictionary:

        if hasattr(self.whole_dataset, 'make_tag_dictionary'):
            return getattr(self.whole_dataset, 'make_tag_dictionary')(tag_type)

        # Make the tag dictionary
        tag_dictionary: Dictionary = Dictionary(add_unk=False)
        tokens = []
        for sentence in self.get_all_sentences():
            for token in sentence.tokens:
                tokens.append(token.get_tag(tag_type).value)
        tokens.sort()
        for token in tokens:
            tag_dictionary.add_item(token)
        return tag_dictionary


class ToyDataset(FlairDataset):
    """
      a   x ... x   b
      b
    """

    def __init__(
            self,
            vocab_length: int = 100,
            tag_set_size: int = 20,
            max_sentence_length: int = 40,
            total_sentence_count: int = 1000,
            label: str = 'toy-tag'
    ):
        self.vocab_length = vocab_length
        self.tag_set_size = tag_set_size
        self.max_sentence_length = max_sentence_length
        self.total_sentence_count = total_sentence_count
        self.label = label

        # generate vocab
        self.vocab = [''.join(random.sample('zyxwvutsrqponmlkjihgfedcba',5)) for _ in range(vocab_length)]
        self.tag_set = [''.join(random.sample('ABCDEFGHIJKLMNOPQRSTUVWXYZ',5)) for _ in range(tag_set_size)]
        

    def is_in_memory(self) -> bool:
        return True

    def __len__(self):
        return self.total_sentence_count

    def __getitem__(self, index: int = 0) -> Sentence:

        sentence = self.gen_sentence(index)

        return sentence
    
    def gen_sentence(self, index: int = 0) -> Sentence:
        
        rand = random.Random(index)

        sentence: Sentence = Sentence()
        sentence_length = rand.choice(range(2, self.max_sentence_length+1))

        token = Token('0_first_word')
        sentence.add_token(token)
        for _ in range(sentence_length-2):
            token = Token(rand.choice(self.vocab))
            # token.add_label(self.label, rand.choice(self.tag_set))
            token.add_label(self.label, token.text)
            sentence.add_token(token)
        token = Token(rand.choice(self.tag_set))
        token.add_label(self.label, token.text)
        sentence.add_token(token)
        sentence[0].add_label(self.label, token.text)
        
        return sentence


class Toy2Dataset(FlairDataset):
    """
      mask   x-x ... x-x  x-s  x-b
      s-b

      mask   x-x ... x-x  x-t  x-b
      t-b

      (x is random)
    """

    def __init__(
            self,
            vocab_length: int = 100,
            tag_set_size: int = 20,
            max_sentence_length: int = 40,
            total_sentence_count: int = 1000,
            label: str = 'toy-tag'
    ):
        self.vocab_length = vocab_length
        self.tag_set_size = tag_set_size
        self.max_sentence_length = max_sentence_length
        self.total_sentence_count = total_sentence_count
        self.label = label

        # generate vocab
        self.tag_set = [''.join(random.sample('ABCDEFGHIJKLMNOPQRSTUVWXYZ',5)) for _ in range(tag_set_size)]
        self.vocab = [a+'-'+b for a in self.tag_set for b in self.tag_set] #+ [''.join(random.sample('zyxwvutsrqponmlkjihgfedcba',5)) for _ in range(vocab_length - tag_set_size**2)]

    def is_in_memory(self) -> bool:
        return True

    def __len__(self):
        return self.total_sentence_count

    def __getitem__(self, index: int = 0) -> Sentence:

        sentence = self.gen_sentence(index)

        return sentence
    
    def gen_sentence(self, index: int = 0) -> Sentence:
        
        rand = random.Random(index)

        sentence: Sentence = Sentence()
        sentence_length = rand.choice(range(3, self.max_sentence_length+1))

        token = Token('<MASK>')
        sentence.add_token(token)
        # token = Token('<SEP>')
        # token.add_label(self.label, '<pad>')
        # sentence.add_token(token)
        for _ in range(sentence_length-3):
            token = Token(rand.choice(self.vocab))
            token.add_label(self.label, '<pad>')
            sentence.add_token(token)
        text1, text2 = rand.choice(self.tag_set), rand.choice(self.tag_set)
        # token1 = Token('n-' + text1)
        token1 = Token(rand.choice(self.tag_set) + '-' + text1)
        token1.add_label(self.label, '<pad>')
        sentence.add_token(token1)
        # token2 = Token('n-' + text2)
        token2 = Token(rand.choice(self.tag_set) + '-' + text2)
        token2.add_label(self.label, '<pad>')
        sentence.add_token(token2)
        sentence[0].add_label(self.label, text1 + '-' + text2)

        # print(sentence)
        # exit()
        
        return sentence


class Toy3Dataset(FlairDataset):
    """
      a random sequence with random label, but the last token
      ... A B ... A <MASK> -> B
    """

    def __init__(
            self,
            vocab_length: int = 100,
            tag_set_size: int = 20,
            max_sentence_length: int = 40,
            total_sentence_count: int = 1000,
            label: str = 'toy-tag'
    ):
        self.vocab_length = vocab_length
        self.tag_set_size = tag_set_size # deprecated
        self.max_sentence_length = max_sentence_length
        self.total_sentence_count = total_sentence_count
        self.label = label
        self.k = 1

        # generate vocab
        self.vocab = [''.join(random.sample('zyxwvutsrqponmlkjihgfedcba',5)) for _ in range(vocab_length)]
        # self.tag_set = [''.join(random.sample('ABCDEFGHIJKLMNOPQRSTUVWXYZ',5)) for _ in range(tag_set_size)]
        self.tag_set = [t.upper() for t in self.vocab]

    def is_in_memory(self) -> bool:
        return True

    def __len__(self):
        return self.total_sentence_count

    def __getitem__(self, index: int = 0) -> Sentence:

        sentence = self.gen_sentence(index)

        return sentence
    
    def gen_sentence(self, index: int = 0) -> Sentence:
        
        rand = random.Random(index)

        sentence: Sentence = Sentence()
        sentence_length = rand.choice(range(3+self.k, self.max_sentence_length+1))

        for _ in range(sentence_length - 2):
            token = Token(rand.choice(self.vocab))
            token.add_label(self.label, '<pad>')
            sentence.add_token(token)

        idx = rand.randint(0, sentence_length-3-self.k)

        token = Token(sentence[idx].text)
        token.add_label(self.label, '<pad>')
        sentence.add_token(token)

        token = Token('<MASK>')
        token.add_label(self.label, sentence[idx+self.k].text)
        sentence.add_token(token)

        # for _ in range(rand.choice(range(3+self.k, self.max_sentence_length+1))):
        #     token = Token(rand.choice(self.vocab))
        #     token.add_label(self.label, '<pad>')
        #     sentence.add_token(token)
        
        # print(sentence)
        # exit()
        
        return sentence


class Toy4Dataset(FlairDataset):
    """
      ... <pad> A ... <MASK> -> A
    """

    def __init__(
            self,
            vocab_length: int = 100,
            tag_set_size: int = 20,
            max_sentence_length: int = 40,
            total_sentence_count: int = 1000,
            label: str = 'toy-tag'
    ):
        self.vocab_length = vocab_length
        self.tag_set_size = tag_set_size # deprecated
        self.max_sentence_length = max_sentence_length
        self.total_sentence_count = total_sentence_count
        self.label = label
        self.k = 5

        # generate vocab
        self.vocab = [''.join(random.sample('zyxwvutsrqponmlkjihgfedcba',5)) for _ in range(vocab_length)]

    def is_in_memory(self) -> bool:
        return True

    def __len__(self):
        return self.total_sentence_count

    def __getitem__(self, index: int = 0) -> Sentence:

        sentence = self.gen_sentence(index)

        return sentence
    
    def gen_sentence(self, index: int = 0) -> Sentence:
        
        rand = random.Random(index)

        sentence: Sentence = Sentence()
        sentence_length = rand.choice(range(2+self.k, self.max_sentence_length+1))

        for _ in range(sentence_length - 1):
            token = Token(rand.choice(self.vocab))
            token.add_label(self.label, '<pad>')
            sentence.add_token(token)

        idx = rand.randint(0, sentence_length-2-self.k)

        sentence[idx].text = '<pad>'

        token = Token('<MASK>')
        token.add_label(self.label, sentence[idx+self.k].text)
        sentence.add_token(token)

        # for _ in range(rand.choice(range(self.max_sentence_length+1))):
        #     token = Token(rand.choice(self.vocab))
        #     token.add_label(self.label, '<pad>')
        #     sentence.add_token(token)

        # print(sentence)
        # exit()
        
        return sentence


class Toy5Dataset(FlairDataset):
    """
      A random sentence repeat 2 times, randomly mask one word
    """

    def __init__(
            self,
            vocab_length: int = 100,
            tag_set_size: int = 20,
            max_sentence_length: int = 40,
            total_sentence_count: int = 1000,
            label: str = 'toy-tag'
    ):
        self.vocab_length = vocab_length
        self.tag_set_size = tag_set_size # deprecated
        self.max_sentence_length = max_sentence_length
        self.total_sentence_count = total_sentence_count
        self.label = label
        self.add_noise = False
        self.with_replacement = True

        # generate vocab
        self.vocab = [''.join(random.sample('zyxwvutsrqponmlkjihgfedcba',5)) for _ in range(vocab_length)]

    def is_in_memory(self) -> bool:
        return True

    def __len__(self):
        return self.total_sentence_count

    def __getitem__(self, index: int = 0) -> Sentence:

        sentence = self.gen_sentence(index)

        return sentence
    
    def gen_sentence(self, index: int = 0) -> Sentence:
        
        rand = random.Random(index)

        sentence: Sentence = Sentence()
        sentence_length = rand.choice(range(2, self.max_sentence_length+1))
        # sentence_length = self.max_sentence_length

        if self.with_replacement:
            words = [rand.choice(self.vocab) for _ in range(sentence_length//2)] * 2
        else:
            words = self.vocab[:]
            rand.shuffle(words)
            words = words[:sentence_length//2] * 2

        idx = rand.randint(0, len(words)-1)

        if self.add_noise and rand.random() > 0.5:
            token = Token(rand.choice(self.vocab))
            token.add_label(self.label, '<pad>')
            sentence.add_token(token)

        for i in range(sentence_length//2*2):
            if i == sentence_length//2 and self.add_noise and rand.random() > 0.5:
                token = Token(rand.choice(self.vocab))
                token.add_label(self.label, '<pad>')
                sentence.add_token(token)

            if i == idx:
                token = Token('<MASK>')
                token.add_label(self.label, words[i])
                sentence.add_token(token)
            else:
                token = Token(words[i])
                token.add_label(self.label, '<pad>')
                sentence.add_token(token)

        if self.add_noise and rand.random() > 0.5:
            token = Token(rand.choice(self.vocab))
            token.add_label(self.label, '<pad>')
            sentence.add_token(token)

        # print(sentence)
        # exit()
        
        return sentence

    def make_tag_dictionary(self, tag_type: str) -> Dictionary:

        # Make the tag dictionary
        tag_dictionary: Dictionary = Dictionary(add_unk=False)
        for token in sorted(self.vocab):
            tag_dictionary.add_item(token)
        return tag_dictionary