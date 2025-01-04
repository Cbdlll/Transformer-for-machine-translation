import torch

def create_padding_mask(seq, pad_token_id):
    """
    创建 Padding Mask，用于屏蔽填充的 token。
    seq: [batch_size, seq_len]
    pad_token_id: 填充 token 的 ID
    返回: [batch_size, 1, 1, seq_len]
    """
    return (seq == pad_token_id).unsqueeze(1).unsqueeze(2)

def create_look_ahead_mask(seq_len):
    """
    创建 Look-Ahead Mask，用于屏蔽未来的信息。
    size: 序列长度
    返回: [seq_len, seq_len]
    """
    mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
    return mask # 1 表示该位置被屏蔽。

def create_combined_mask(seq, pad_token_id, device=None):
    """
    创建 Combined Mask，结合 Padding Mask 和 Look-Ahead Mask。

    参数：
    - seq: [batch_size, seq_len]，输入序列
    - pad_token_id: 填充 token 的 ID

    返回：
    - combined_mask: [batch_size, 1, seq_len, seq_len]，联合掩码
    """
    # 1. 创建 Padding Mask
    padding_mask = create_padding_mask(seq, pad_token_id).to(device)

    # 2. 创建 Look-Ahead Mask
    look_ahead_mask = create_look_ahead_mask(seq.size()[1]).to(device)

    # 3. 合并两个 Mask
    combined_mask = look_ahead_mask.unsqueeze(0).unsqueeze(0) | padding_mask  # [batch_size, 1, seq_len, seq_len]

    return combined_mask