import torch
import torch.nn.functional as F
import sacrebleu
from transformer import Transformer
from simpleTokenizer import SimpleTokenizer

def translate_sentence(model, src_sentence, src_tokenizer, tgt_tokenizer, max_len, device):
    """
    使用训练好的 Transformer 模型翻译单个句子。
    """
    # 将源句子转换为张量
    src_input = src_tokenizer.encode(src_sentence, max_length=max_len, padding=True, add_bos_eos=True).unsqueeze(0).to(device)
    
    # 构建 Padding Mask
    src_padding_mask = (src_input == src_tokenizer.pad_id).unsqueeze(1).unsqueeze(2).to(device)

    # 编码器前向传播
    enc_output = model.encoder(src_input, src_padding_mask)

    # 解码器初始输入 [<bos>]
    decoder_input = torch.tensor([[tgt_tokenizer.bos_id]], device=device)

    generated_tokens = []

    for _ in range(max_len):
        
        # Look-Ahead Mask
        tgt_look_ahead_mask = torch.triu(torch.ones(decoder_input.size(1), decoder_input.size(1), device=device, dtype=torch.bool), diagonal=1)
        
        # 解码器前向传播
        dec_output = model.decoder(decoder_input, enc_output, tgt_look_ahead_mask, src_padding_mask)

        # 线性层 + Softmax 预测下一个词
        logits = model.out_linear(dec_output)
        probs = F.softmax(logits[:, -1, :], dim=-1)
        next_token = probs.argmax(dim=-1)  # Greedy 解码
        # next_token = torch.multinomial(probs, num_samples=1)  # 随机采样
   
        # 如果生成 <eos>，停止
        if next_token.item() == tgt_tokenizer.eos_id:
            break
        # 添加到已生成序列中
        generated_tokens.append(next_token.item())
        decoder_input = torch.cat([decoder_input, next_token.unsqueeze(0)], dim=1)
        # decoder_input = torch.cat([decoder_input, next_token], dim=1)  # 随机采样

    # 将生成的 Token 转换为文本
    return join_tokens(tgt_tokenizer.decode(generated_tokens))

def join_tokens(tokens):
    sentence = ""
    for token in tokens:
        if token in {',', '.', '!', '?', ';', ':', '(', ')', '[', ']', '{', '}', '<PAD>'}:
            sentence += token  # 符号前面不加空格
        else:
            sentence += " " + token  # 非符号前面加空格
    return sentence.strip()  # 去掉开头和结尾的空格

def evaluate_model(model, test_data, src_tokenizer, tgt_tokenizer, max_len, device):
    model.eval()  # 设置为评估模式
    total_bleu = 0

    with torch.no_grad():
        for src_sentence, tgt_sentence in test_data:
            # 翻译源句子
            prediction = translate_sentence(model, src_sentence, src_tokenizer, tgt_tokenizer, max_len, device)
            # 计算 BLEU 分数
            bleu = sacrebleu.sentence_bleu(prediction, [tgt_sentence]).score
            total_bleu += bleu
            print(f"Source: {src_sentence}")
            print(f"Target: {tgt_sentence}")
            print(f"Prediction: {prediction}")
            print(f"BLEU: {bleu:.2f}")

        # 返回平均 BLEU 分数
        return total_bleu / len(test_data)


if __name__ == '__main__':
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Transformer(3886, 3366, 512, 8, 6, 256, 0.1).to(device)
    # model.load_state_dict(torch.load('transformer_translation.pth', map_location=torch.device('cpu'), weights_only=True))
    model.load_state_dict(torch.load('transformer_translation.pth', weights_only=True))

    tokenizer_src = SimpleTokenizer(vocab_path='vocab_1000/vocab_zh.json')
    tokenizer_tgt = SimpleTokenizer(vocab_path='vocab_1000/vocab_en.json')
    
    src_sentence = '海洋是一个非常复杂的事物。'
    tgt_sentence = 'It can be a very complicated thing, the ocean.'

    # 计算 BLEU 分数
    evaluate_model(model, [(src_sentence, tgt_sentence)], tokenizer_src, tokenizer_tgt, 128, device)