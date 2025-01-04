from simpleTokenizer import SimpleTokenizer

def calculate_lengths(file_path):

  if file_path.endswith(".en"):
    tokenizer = SimpleTokenizer(vocab_path='vocab/vocab_en.json')
  elif file_path.endswith(".zh"):
    tokenizer = SimpleTokenizer(vocab_path='vocab/vocab_zh.json')
  else:
    raise ValueError("Unsupported file extension. Expected '.en' or '.zh'")


  lengths = []
  with open(file_path, 'r', encoding='utf-8') as f:
    for line in f:
      line = line.strip()
      input_ids = tokenizer.encode(line, add_bos_eos=True)
      lengths.append(len(input_ids))

  # 对长度列表进行降序排序，并取前三个
  lengths.sort(reverse=True)
  return lengths[:100]

# 示例用法
train_zh_file = "data_1000/train/train.zh"
train_en_file = "data_1000/train/train.en"
lengths_zh = calculate_lengths(train_zh_file)
lengths_en = calculate_lengths(train_en_file)

print("ZH序列的长度:", lengths_zh)
print("EN序列的长度:", lengths_en)