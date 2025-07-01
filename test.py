import argparse
import os

import numpy as np
import torch
import yaml
from torch.nn.functional import cosine_similarity

from utils.model import CBOW_Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_pytorch_model(vocab_path, model_path):
    vocab = torch.load(vocab_path,  map_location=device)
    model = torch.load(model_path, map_location=device)
    model.eval()

    # 从加载的model中获取其嵌入层（embeddings）的权重参数（即词向量矩阵），并通过.detach()方法将其从计算图中分离（不再进行梯度计算）
    # 再用.to(device)将其移动到指定的设备（如GPU或CPU）上，最终得到一个可用于后续相似度计算的词向量张量。
    embeddings = model.embeddings.weight.detach().to(device)
    return vocab, embeddings

def calculate_similarity(word1, word2, vocab_dict, embeddings):
    def get_word_vector(word):
        if word in vocab_dict:
            # 根据传入的单词 word，先通过 vocab_dict[word] 获取该单词在词汇表中的索引，
            # 然后用这个索引从 embeddings（词向量矩阵）中取出对应的词向量。
            # unsqueeze(0) 的作用是给这个词向量增加一个批次维度，使其形状从 [embedding_dim] 变为 [1, embedding_dim]，
            # 方便后续与其他词向量进行批量计算（如余弦相似度）。
            return embeddings[vocab_dict[word]].unsqueeze(0)
        else:
            return torch.zeros((1, embeddings.size(1)), device=device)
    vec1 = get_word_vector(word1)
    # print(vec1)
    vec2 = get_word_vector(word2)
    # print(vec2)
    # 计算两个词向量 vec1 和 vec2 之间的余弦相似度，并通过 .item() 方法将结果从张量转换为 Python 浮点数。
    # 余弦相似度用于衡量两个向量在向量空间中的相似程度，值域为 [-1, 1]，值越大表示越相似。
    similarity = cosine_similarity(vec1, vec2).item()
    return similarity

def test(config):
    word_pairs = [
        ("father", "mother"),
        ("king", "queen"),
        ("apple", "orange"),
        ("machine", "learning"),
        ("book", "car")
    ]
    vocab_path = os.path.join(config["model_dir"], "vocab.pt")
    model_path = os.path.join(config["model_dir"], "model.pt")
    vocab_dict, embeddings = load_pytorch_model(vocab_path, model_path)

    for word1, word2 in word_pairs:
        sim = calculate_similarity(word1, word2, vocab_dict, embeddings)
        print(f"词语对 '{word1}' 和 '{word2}' 的相似度: {sim:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='path to yaml config')
    args = parser.parse_args()

    with open(args.config, 'r') as stream:
        config = yaml.safe_load(stream)
    try:
        test(config)
    except Exception as e:
        print(f"运行出错: {e}")