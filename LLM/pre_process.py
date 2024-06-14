import torch
import torch.nn as nn

# 假设我们有一个简单的文本数据
text_data = ["I love eating apples", "Bananas are my favorite fruit", "when I was a young man, I only ate bananas and apples"]

# 简单的分词函数，这里只是按照空格分词
def tokenize(text):
    return text.lower().split()

# 创建词汇表
def create_vocab(texts):
    vocab = {
        '<PAD>': 0,  # 填充标记固定为索引0
        '<UNK>': 1   # 未知标记固定为索引1
    }
    for text in texts:
        tokens = tokenize(text)
        for token in tokens:
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab

# 将文本转换为索引序列，并添加填充
def text_to_indices(text, word_to_index, max_length):
    tokens = tokenize(text)
    indexed_line = [word_to_index.get(token, word_to_index['<UNK>']) for token in tokens[:max_length]]
    # 如果句子长度小于max_length，用<PAD>填充
    indexed_line += [word_to_index['<PAD>']] * (max_length - len(indexed_line))
    return indexed_line[:max_length]

# 创建embedding层
def create_embedding_layer(vocab_size, embedding_dim):
    return nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

# 确定最大句子长度
max_length = max(len(tokenize(text)) for text in text_data)

# 创建词汇表
vocab = create_vocab(text_data)
word_to_index = vocab

# 创建embedding层
vocab_size = len(word_to_index)  # 更新词汇表大小
embedding_dim = 10  # 可以调整embedding维度
embedding_layer = create_embedding_layer(vocab_size, embedding_dim)

test_data = ["I love eating bananas", "I love eating apples and bananas", "I love eating apples, bananas, and oranges"]
# 将文本数据转换为索引序列，并添加填充
indexed_data = [text_to_indices(text, word_to_index, max_length) for text in test_data]

# 将索引序列转换为张量，以便使用embedding层
tensor_data = torch.tensor(indexed_data)

# 打印转换后的张量
print("Indexed and Padded Data Tensor:")
print(tensor_data)

# 使用embedding层获取词向量
word_embeddings = embedding_layer(tensor_data)
