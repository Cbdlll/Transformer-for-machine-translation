import torch
import json
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from simpleTokenizer import SimpleTokenizer

def load_vocab(vocab_file):
    with open(vocab_file, 'r', encoding='utf-8') as f:
        return json.load(f)
    
    
class TranslationDataset(Dataset):
    def __init__(self, zh_file, en_file, zh_tokenizer, en_tokenizer, max_len=512):
        self.zh_tokenizer = zh_tokenizer
        self.en_tokenizer = en_tokenizer
        self.max_len = max_len
        
        # 加载文件
        with open(zh_file, "r", encoding="utf-8") as f:
            self.zh_lines = [line.strip() for line in f.readlines()]
        with open(en_file, "r", encoding="utf-8") as f:
            self.en_lines = [line.strip() for line in f.readlines()]
        
    def __len__(self):
        return len(self.zh_lines)
    
    def __getitem__(self, idx):
        zh_text = self.zh_lines[idx]
        en_text = self.en_lines[idx]
        
        zh_tokens = self.zh_tokenizer.encode(zh_text, self.max_len, truncation=True, padding=True, add_bos_eos=False)
        en_tokens = self.en_tokenizer.encode(en_text, self.max_len, truncation=True, padding=True, add_bos_eos=True)

        return zh_tokens, en_tokens
    
    
class TranslationDataLoader(DataLoader):
    def __init__(self, train_zh_file, train_en_file, zh_tokenizer, en_tokenizer, max_len=512, batch_size=32, shuffle=True):
        dataset = TranslationDataset(train_zh_file, train_en_file, zh_tokenizer, en_tokenizer, max_len)
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.collate_fn
        )

    def collate_fn(self, batch):
        src_inputs = torch.stack([i[0] for i in batch], dim=0)  # i[0] 是源语言
        tgt_inputs = torch.stack([i[1] for i in batch], dim=0)  # i[1] 是目标语言

        return src_inputs, tgt_inputs

if __name__ == '__main__':
    
    # 加载英文和中文分词器
    zh_tokenizer = SimpleTokenizer(vocab_path='vocab/vocab_zh.json')
    en_tokenizer = SimpleTokenizer(vocab_path='vocab/vocab_en.json')
    
    train_zh_file = "data/train/train.zh"
    train_en_file = "data/train/train.en"

    # dataset = TranslationDataset(train_zh_file, train_en_file, zh_tokenizer, en_tokenizer, max_len=512)
    # data_loader = DataLoader(dataset, batch_size=32, shuffle=True,     
    #     collate_fn=lambda x: (
    #     torch.stack([torch.tensor(i[0]) for i in x], dim=0),  # src_input
    #     torch.stack([torch.tensor(i[1]) for i in x], dim=0)   # tgt_input
    # ))
    
    data_loader = TranslationDataLoader(train_zh_file, train_en_file, zh_tokenizer, en_tokenizer, max_len=512, batch_size=32, shuffle=True)
    
    # 获取一个批次的数据
    batch = next(iter(data_loader))

    zh_batch, en_batch = batch  # en_batch 和 zh_batch 分别是英文和中文的批次
    
    # print(type(batch))
    # print(type(zh_batch))
    # print(zh_batch)

    # 查看第一个样本
    zh_sample = zh_batch[0]
    en_sample = en_batch[0]

    print("First Chinese sample:", zh_sample)
    print(zh_sample.shape)
    print(zh_tokenizer.decode(zh_sample.tolist()))
    print("First English sample:", en_sample)
    print(en_tokenizer.decode(en_sample.tolist()))