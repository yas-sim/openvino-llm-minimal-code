import os
import re
import subprocess
import numpy as np
import logging

import json

logging.basicConfig(level=logging.INFO)

class SimpleTokenizer:
    def __init__(self, model_vendor, model_name):
        self.logger = logging.getLogger(__name__)
        self.logger.info('You are using \'SimpleTokenizer\' which is low-performance.')
        self.tokenizer_dir = f'./{model_vendor}--{model_name}--tokenizer'
        # Download data files
        if not os.path.exists(self.tokenizer_dir):
            subprocess.call(f'huggingface-cli download --local-dir={self.tokenizer_dir} {model_vendor}/{model_name} tokenizer.json tokenizer_config.json special_tokens_map.json')

        with open(f'{self.tokenizer_dir}/tokenizer.json', encoding='utf8') as f:
            self.tokenizer_json = json.load(f)
        vocab_orig = self.tokenizer_json['model']['vocab']                          # Obtain the vocaburary dictionary
        vocab = dict()
        self.decode_list = [''] * len(vocab_orig)
        for key, val in vocab_orig.items():                                         # Do some translations and conversions
            key = key.replace('▁', ' ')
            m = re.match(r'^<(0x[0-9A-F]{2})>$', key)                               # Find "<0xXX>"
            if m is not None:
                hex_val = m.groups()[0]
                key = chr(int(hex_val, 16))                                         # "<0xXX>" -> charactor
            vocab[key] = val
            self.decode_list[val] = key                                             # list for decoding
        vocab = sorted(vocab.items(), key=lambda x:len(x[0]), reverse=True)         # Sort by the length of the keyword
        self.vocab = dict((x, y) for x, y in vocab)                                 # Convert back to dict
        self.num_vocab = len(self.vocab)

        with open(f'{self.tokenizer_dir}/special_tokens_map.json', encoding='utf8') as f:
            self.special_tokens_map_json = json.load(f)

        def get_token_info(token_name:str) -> tuple[str, int]:
            token_str = self.special_tokens_map_json.get(token_name)
            if token_str is None:
                return None, 0
            if isinstance(token_str, dict):
                token_str = token_str['content']
            return (token_str, self.vocab[token_str])

        self.bos_token, self.bos_token_id = get_token_info('bos_token')
        self.eos_token, self.eos_token_id = get_token_info('eos_token')
        self.pad_token, self.pad_token_id = get_token_info('pad_token')
        self.unk_token, self.unk_token_id = get_token_info('unk_token')

    def encode(self, text:str) -> list[int]:
        token_ids = []
        while len(text) > 0:
            token_id = self.unk_token_id
            for key, val in self.vocab.items():
                if text[:len(key)] == key:
                    token_id = val
                    text = text[len(key):]
                    break
            if token_id == self.unk_token_id:
                text = text[1:]
            token_ids.append(token_id)

        self.input_ids = [token_ids]
        self.attention_mask = [[ 1 for _ in range(len(token_ids)) ]]
        self.position_ids = [range(len(token_ids))]
        self.beam_idx = [0]
        return self.input_ids

    def __call__(self, text:str, return_tensors:str='np'):
        self.encode(text)

        if return_tensors == 'np':
            self.input_ids      = np.array(self.input_ids, dtype=np.int64)
            self.attention_mask = np.array(self.attention_mask, dtype=np.int64)
            self.position_ids   = np.array(self.position_ids, dtype=np.int64)
            self.beam_idx       = np.array(self.beam_idx)
        if return_tensors == 'pt':
            raise NotImplementedError
            #import torch
            #self.input_ids      = torch.Tensor(self.input_ids)
            #self.attention_mask = torch.Tensor(self.attention_mask)
            #self.position_ids   = torch.Tensor(self.position_ids)
            #self.beam_idx       = torch.Tensor(self.beam_idx)
        return self

    def decode(self, token_ids:list[int]) -> str:
        text = ''
        for token_id in token_ids:
            text += self.decode_list[token_id]
        return text
