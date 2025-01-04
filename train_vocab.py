import os
import jieba
import logging
from nltk.tokenize import word_tokenize
from collections import Counter
import json
jieba.setLogLevel(logging.WARNING)

# 分词和统计词频
def tokenize_and_build_vocab(input_file, output_file, vocab_file, min_freq=1):
    """
    对中文文本进行分词，并生成词汇表
    :param input_file: 输入的原始中文文本文件路径
    :param output_file: 分词后的输出文件路径
    :param vocab_file: 词汇表保存路径
    :param min_freq: 最小词频，小于该频率的词将被丢弃
    """
    tokenized_sentences = []
    word_counter = Counter()
    
    # 分词
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if input_file.endswith(".zh"):
                tokens = list(jieba.cut(line.strip()))
            else: 
                tokens = list(word_tokenize(line.strip()))
            tokenized_sentences.append(' '.join(tokens))
            word_counter.update(tokens)
    
    # 保存分词结果
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for sentence in tokenized_sentences:
            f.write(sentence + '\n')
    
    # 构建词汇表
    vocab = {'<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3}  # 预留特殊标记
    for word, freq in word_counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    
    # 保存词汇表
    os.makedirs(os.path.dirname(vocab_file), exist_ok=True)
    with open(vocab_file, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    
    print(f"分词结果已保存到: {output_file}")
    print(f"词汇表已保存到: {vocab_file}, 共计 {len(vocab)} 个词汇.")
    

if __name__ == '__main__':
    
    # zh_train_file = "data/train/train.zh"
    # en_train_file = "data/train/train.en"
    
    # zh_tokenized_file = "tokenized/tokenized.zh"
    # en_tokenized_file = "tokenized/tokenized.en"
    
    # zh_vocab_file = "vocab/vocab_zh.json"
    # en_vocab_file = "vocab/vocab_en.json"
    
    zh_train_file = "data_1000/train/train.zh"
    en_train_file = "data_1000/train/train.en"
    
    zh_tokenized_file = "tokenized_1000/tokenized.zh"
    en_tokenized_file = "tokenized_1000/tokenized.en"
    
    zh_vocab_file = "vocab_1000/vocab_zh.json"
    en_vocab_file = "vocab_1000/vocab_en.json"
    
    tokenize_and_build_vocab(zh_train_file, zh_tokenized_file, zh_vocab_file)
    tokenize_and_build_vocab(en_train_file, en_tokenized_file, en_vocab_file)