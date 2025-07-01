import torch.nn as nn

from utils.constants import EMBED_DIMENSION, EMBED_MAX_NORM


# 初始化CBOW模型
class CBOW_Model(nn.Module):
    """
    Implementation of CBOW model described in paper:
    https://arxiv.org/abs/1301.3781
    """
    def __init__(self, vocab_size: int):
        super(CBOW_Model, self).__init__()
        # 嵌入层，将词ID映射为固定维度的向量，维度由EMBED_DIMENSION指定，max_norm用于正则化嵌入向量的范数。
        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=EMBED_DIMENSION,
            max_norm=EMBED_MAX_NORM,
        )
        # 线性层，将嵌入向量映射回词表大小的输出，用于预测目标词。
        self.linear = nn.Linear(
            in_features=EMBED_DIMENSION,
            out_features=vocab_size,
        )

    # 向前传播
    def forward(self, inputs_):
        # 先通过嵌入层得到每个词的向量表示
        x = self.embeddings(inputs_)
        # 对上下文词的向量在第1维（即词的维度）做平均，得到上下文的整体表示。
        x = x.mean(axis=1)
        # 通过线性层输出预测结果，通常用于softmax分类。
        x = self.linear(x)
        return x


class SkipGram_Model(nn.Module):
    """
    Implementation of Skip-Gram model described in paper:
    https://arxiv.org/abs/1301.3781
    """
    def __init__(self, vocab_size: int):
        super(SkipGram_Model, self).__init__()
        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=EMBED_DIMENSION,
            max_norm=EMBED_MAX_NORM,
        )
        self.linear = nn.Linear(
            in_features=EMBED_DIMENSION,
            out_features=vocab_size,
        )

    def forward(self, inputs_):
        x = self.embeddings(inputs_)
        x = self.linear(x)
        return x