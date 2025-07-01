import argparse
import yaml
import os
import torch
import torch.nn as nn

from utils.dataloader import get_dataloader_and_vocab
from utils.trainer import Trainer
from utils.helper import (
    get_model_class,
    get_optimizer_class,
    get_lr_scheduler,
    save_config,
    save_vocab,
)


def train(config):
    os.makedirs(config["model_dir"])

    # 下载数据集，初始化训练集数据加载器，并初始化词表
    train_dataloader, vocab = get_dataloader_and_vocab(
        model_name=config["model_name"],
        ds_name=config["dataset"],
        ds_type="train",
        data_dir=config["data_dir"],
        batch_size=config["train_batch_size"],
        shuffle=config["shuffle"],
        vocab=None,
    )

    # 初始化验证集数据加载器
    val_dataloader, _ = get_dataloader_and_vocab(
        model_name=config["model_name"],
        ds_name=config["dataset"],
        ds_type="valid",
        data_dir=config["data_dir"],
        batch_size=config["val_batch_size"],
        shuffle=config["shuffle"],
        vocab=vocab,
    )

    # 获取词表大小
    vocab_size = len(vocab.get_stoi())
    print(f"Vocabulary size: {vocab_size}")

    # 初始化模型
    model_class = get_model_class(config["model_name"])
    model = model_class(vocab_size=vocab_size)
    # 定义损失函数为交叉熵损失，常用于分类任务
    criterion = nn.CrossEntropyLoss()

    # 根据配置文件获取优化器类
    optimizer_class = get_optimizer_class(config["optimizer"])
    # 用指定学习率初始化优化器，优化模型参数。
    optimizer = optimizer_class(model.parameters(), lr=config["learning_rate"])
    # 为优化器设置学习率调度器，动态调整学习率。
    lr_scheduler = get_lr_scheduler(optimizer, config["epochs"], verbose=True)

    # 自动检测是否有可用GPU，有则用GPU，否则用CPU。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainer = Trainer(
        model=model,
        epochs=config["epochs"],
        train_dataloader=train_dataloader,
        train_steps=config["train_steps"],
        val_dataloader=val_dataloader,
        val_steps=config["val_steps"],
        criterion=criterion,
        optimizer=optimizer,
        checkpoint_frequency=config["checkpoint_frequency"],
        lr_scheduler=lr_scheduler,
        device=device,
        model_dir=config["model_dir"],
        model_name=config["model_name"],
    )

    # 开始训练
    trainer.train()
    print("Training finished.")

    # 保存模型、损失函数、词表、config
    trainer.save_model()
    trainer.save_loss()
    save_vocab(vocab, config["model_dir"])
    save_config(config, config["model_dir"])
    print("Model artifacts saved to folder:", config["model_dir"])
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='path to yaml config')
    args = parser.parse_args()
    
    with open(args.config, 'r') as stream:
        config = yaml.safe_load(stream)
    train(config)