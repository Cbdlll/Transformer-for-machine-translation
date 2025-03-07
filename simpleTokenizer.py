import torch
import json
import jieba
import logging
from nltk.tokenize import word_tokenize
jieba.setLogLevel(logging.WARNING)

class SimpleTokenizer:
    def __init__(self, vocab_path):
        self.vocab_path = vocab_path
        # 加载词汇表
        try:
            with open(self.vocab_path, "r", encoding="utf-8") as f:
                    self.vocab = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"词汇表文件 {self.vocab_path} 未找到")
        except json.JSONDecodeError:
            raise json.JSONDecodeError(f"词汇表文件 {self.vocab_path} 格式错误")
        
        self.id_to_token = {v: k for k, v in self.vocab.items()}  # 创建反向词汇表，便于解码
        
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
        
        self.pad_id = self.vocab[self.pad_token]
        self.unk_id = self.vocab[self.unk_token]
        self.bos_id = self.vocab[self.bos_token]
        self.eos_id = self.vocab[self.eos_token]
        
    def tokenize(self, text):
        # 假设文本是由空格分隔的词汇，可以用其他分词工具替代
        if 'zh' in self.vocab_path:
            tokens = list(jieba.cut(text.strip()))
        else:
            tokens = list(word_tokenize(text.strip()))
            
        return tokens
    
    
    def encode(self, text, max_length=512, truncation=False, padding=False, add_bos_eos=False):
        tokens = self.tokenize(text)
        # tokens = list[]
        token_ids = [self.vocab.get(token, self.unk_id) for token in tokens]
        
        # 加入 <BOS> 和 <EOS> 标记
        if add_bos_eos:
            token_ids = [self.bos_id] + token_ids + [self.eos_id]
        
        # 截断
        if truncation and len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
        
        # 填充
        if padding and len(token_ids) < max_length:
            token_ids += [self.pad_id] * (max_length - len(token_ids))
        
        return torch.tensor(token_ids, dtype=torch.long)
    
    def decode(self, token_ids, skip_special_tokens=False):
        decoded_tokens = []
        for token_id in token_ids:
            token = self.id_to_token.get(token_id, self.unk_token)
            if skip_special_tokens and token in {self.pad_token, self.unk_token, self.bos_token, self.eos_token}:
                continue
            decoded_tokens.append(token)
        return decoded_tokens
    
if __name__ == '__main__':
    
    zh_tokenzier = SimpleTokenizer(vocab_path='vocab/vocab_zh.json')
    zh_sentence = "接下来如果我们能理解这一点 从这里出发， 可以得出下一步的， 那就是如果海洋不高兴了 大家也都别想开心。"
    zh_indices = zh_tokenzier.encode(zh_sentence,add_bos_eos=True)
    print(zh_tokenzier.tokenize(zh_sentence))
    print(zh_indices)
    print(zh_tokenzier.decode(zh_indices.tolist()))
    print(len(zh_indices))
    
    
    # en_tokenzier = SimpleTokenizer(vocab_path='vocab/vocab_en.json')
    # en_sentence = "And if we just take that and we build from there, then we can go to the next step, which is that if the ocean happy, nobody happy"
    # en_indices = en_tokenzier.encode(en_sentence, add_bos_eos=True)
    # print(en_tokenzier.tokenize(en_sentence))
    # print(en_indices)
    # print(en_tokenzier.decode(en_indices.tolist()))
    # print(len(en_indices))
    