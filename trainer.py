import torch
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from mask import create_padding_mask, create_combined_mask
from transformer import Transformer
from simpleTokenizer import SimpleTokenizer
from data_preparation import TranslationDataLoader
from label_smooth import LabelSmoothingLoss
from tqdm import tqdm

class Trainer:
    def __init__(self, model, train_data_loader, tokenizer_src, tokenizer_tgt, device=None, num_epochs=5, lr=5e-5, smoothing=0.1):
        self.model = model
        self.train_data_loader = train_data_loader
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.device = device if device else "cpu"
        self.num_epochs = num_epochs
        self.lr = lr
        self.optimizer = AdamW(self.model.parameters(), lr=self.lr)
        # self.loss_fn = CrossEntropyLoss(ignore_index=self.tokenizer_tgt.pad_id)
        self.loss_fn = LabelSmoothingLoss(classes=3366, smoothing=smoothing, ignore_index=self.tokenizer_tgt.pad_id)
        # 将模型移至设备
        self.model.to(self.device)

    def train(self):
        self.model.train()
        total_loss = 0

        for epoch in range(self.num_epochs):
            epoch_loss = 0
            train_loop = tqdm(self.train_data_loader, desc=f"Epoch {epoch + 1}/{self.num_epochs}")
            for src_input, tgt_input in train_loop:
                src_input, tgt_input = src_input.to(self.device), tgt_input.to(self.device)
                
                # Teacher Forcing
                targ_inp = tgt_input[:, :-1]  # 目标输入
                targ_real = tgt_input[:, 1:]  # 目标真实标签

                # 创建 Mask
                padding_mask = create_padding_mask(src_input, self.tokenizer_src.pad_id).to(self.device)
                combined_mask = create_combined_mask(targ_inp, self.tokenizer_tgt.pad_id, self.device).to(self.device)

                # 前向传播
                outputs = self.model(
                    src_input,
                    targ_inp,
                    src_padding_mask=padding_mask,
                    tgt_combined_mask=combined_mask
                )

                # 计算损失
                logits = outputs.reshape(-1, outputs.size(-1))
                labels = targ_real.reshape(-1)
                loss = self.loss_fn(logits, labels)

                # 反向传播和优化
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                train_loop.set_postfix(loss=epoch_loss / (train_loop.n + 1))  # 平均损失

            total_loss += epoch_loss
            print(f"Epoch {epoch + 1}/{self.num_epochs}, Loss: {epoch_loss / len(self.train_data_loader)}")

        print(f"Training completed. Total Loss: {total_loss / (len(self.train_data_loader) * self.num_epochs)}")

    def save_model(self, save_path):
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved at {save_path}")

    def load_model(self, load_path):
        self.model.load_state_dict(torch.load(load_path))
        self.model.to(self.device)
        print(f"Model loaded from {load_path}")


if __name__ == '__main__':
    # 假设您已经定义了模型和 DataLoader
    # (self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, dropout=0.1)
    # model = Transformer(74348, 56918, 512, 8, 6, 256, 0.1)
    model = Transformer(3886, 3366, 512, 8, 6, 256, 0.1)
    tokenizer_src = SimpleTokenizer(vocab_path='vocab_1000/vocab_zh.json')
    tokenizer_tgt = SimpleTokenizer(vocab_path='vocab_1000/vocab_en.json')

    train_zh_file = "data_1000/train/train.zh"
    train_en_file = "data_1000/train/train.en"
    
    # tokenizer_src = SimpleTokenizer(vocab_path='vocab/vocab_zh.json')
    # tokenizer_tgt = SimpleTokenizer(vocab_path='vocab/vocab_en.json')

    # train_zh_file = "data/train/train.zh"
    # train_en_file = "data/train/train.en"
    
    # data_loader = TranslationDataLoader(train_zh_file, train_en_file, tokenizer_src, tokenizer_tgt, max_len=512, batch_size=10, shuffle=True)
    data_loader = TranslationDataLoader(train_zh_file, train_en_file, tokenizer_src, tokenizer_tgt, max_len=128, batch_size=10, shuffle=True)

    # 初始化 Trainer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(model, data_loader, tokenizer_src, tokenizer_tgt, device=device, num_epochs=5)

    # 开始训练
    trainer.train()

    # 保存模型
    trainer.save_model("transformer_translation.pth")

    # 加载模型（如果需要继续训练或评估）
    # trainer.load_model("transformer_translation.pth")
