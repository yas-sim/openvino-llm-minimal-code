import os
import subprocess

import json

class tokenizer:
    def __init__(self, model_vendor, model_name):
        self.tokenizer_dir = f'./{model_vendor}--{model_name}--tokenizer'
        if not os.path.exists(self.tokenizer_dir):
            subprocess.call(f'huggingface-cli download --local-dir={self.tokenizer_dir} {model_vendor}/{model_name} tokenizer.json tokenizer_config.json special_tokens_map.json')

        with open(f'{self.tokenizer_dir}/tokenizer.json', encoding='utf8') as f:
            self.tokenizer_json = json.load(f)
        vocab = self.tokenizer_json['model']['vocab']
        vocab = sorted(vocab.items(), key=lambda x:len(x[0]), reverse=True)
        self.vocab = dict((x.replace('▁', ' '), y) for x, y in vocab)
        self.num_vocab = len(self.vocab)

        self.unk_token = self.tokenizer_json['model']['unk_token']
        self.unk_token_id = self.vocab[self.unk_token]

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
        return token_ids
        
    def decode(self, token_ids:list[int]) -> str:
        text = ''
        for token_id in token_ids:
            for key, val in self.vocab.items():
                if val == token_id:
                    text += key
        return text


class llama_tokenizer(tokenizer):
    def __init__(self):
        super().__init__('TinyLlama', 'TinyLlama-1.1B-Chat-v1.0')
