import os
import numpy as np
import json
import torch


class Trainer:
    """Main class for model training"""
    
    def __init__(
        self,
        model,
        epochs,
        train_dataloader,
        train_steps,
        val_dataloader,
        val_steps,
        checkpoint_frequency,
        criterion,
        optimizer,
        lr_scheduler,
        device,
        model_dir,
        model_name,
    ):  
        self.model = model
        self.epochs = epochs
        self.train_dataloader = train_dataloader
        self.train_steps = train_steps
        self.val_dataloader = val_dataloader
        self.val_steps = val_steps
        self.criterion = criterion
        self.optimizer = optimizer
        self.checkpoint_frequency = checkpoint_frequency
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.model_dir = model_dir
        self.model_name = model_name

        self.loss = {"train": [], "val": []}
        self.model.to(self.device)

    def train(self):
        for epoch in range(self.epochs):
            # 训练
            self._train_epoch()
            self._validate_epoch()
            # 检验
            print(
                "Epoch: {}/{}, Train Loss={:.5f}, Val Loss={:.5f}".format(
                    epoch + 1,
                    self.epochs,
                    self.loss["train"][-1],
                    self.loss["val"][-1],
                )
            )

            # 调用学习率调度器 self.lr_scheduler.step() 更新学习率
            self.lr_scheduler.step()

            # 如果设置了保存检查点的频率（self.checkpoint_frequency），则按频率保存模型检查点
            if self.checkpoint_frequency:
                self._save_checkpoint(epoch)

    def _train_epoch(self):
        # 将模型设置为训练模式，启用如Dropout等训练特性。
        self.model.train()
        # 用于记录每个batch的损失。
        running_loss = []

        for i, batch_data in enumerate(self.train_dataloader, 1):
            # 将输入和标签转移到指定设备（如GPU）
            inputs = batch_data[0].to(self.device)
            labels = batch_data[1].to(self.device)

            # 优化器梯度清零。
            self.optimizer.zero_grad()
            # 前向传播得到输出。
            outputs = self.model(inputs)
            # 计算损失。
            loss = self.criterion(outputs, labels)
            # 反向传播计算梯度。
            loss.backward()
            # 优化器更新参数。
            self.optimizer.step()

            # 记录当前batch的损失。
            running_loss.append(loss.item())

            # 达到指定步数train_steps后提前结束本epoch。
            if i == self.train_steps:
                break

        # 计算本epoch所有batch的平均损失，并保存到self.loss["train"]中。
        epoch_loss = np.mean(running_loss)
        self.loss["train"].append(epoch_loss)

    def _validate_epoch(self):
        # 将模型设置为评估模式，以禁用 dropout 和 batchnorm 等训练特性
        self.model.eval()
        # 用于记录每个 batch 的损失
        running_loss = []

        # 使用 torch.no_grad() 上下文，关闭梯度计算，提高验证效率并节省显存
        with torch.no_grad():
            for i, batch_data in enumerate(self.val_dataloader, 1):
                # 将输入和标签移动到指定设备（如 GPU）
                # cbow模型，输出就是中心词，输入就是上下文（tokenid，向量表示）
                inputs = batch_data[0].to(self.device)
                labels = batch_data[1].to(self.device)

                # 前向传播计算模型输出
                outputs = self.model(inputs)
                # 用损失函数计算损失
                loss = self.criterion(outputs, labels)

                # 损失值添加到 running_loss 列表
                running_loss.append(loss.item())

                # 如果已达到指定的验证步数（self.val_steps），则提前结束循环
                if i == self.val_steps:
                    break

        # 计算本epoch所有batch的平均损失，并保存到self.loss["train"]中。
        epoch_loss = np.mean(running_loss)
        self.loss["val"].append(epoch_loss)

    def _save_checkpoint(self, epoch):
        """Save model checkpoint to `self.model_dir` directory"""
        epoch_num = epoch + 1
        if epoch_num % self.checkpoint_frequency == 0:
            model_path = "checkpoint_{}.pt".format(str(epoch_num).zfill(3))
            model_path = os.path.join(self.model_dir, model_path)
            torch.save(self.model, model_path)

    def save_model(self):
        """Save final model to `self.model_dir` directory"""
        model_path = os.path.join(self.model_dir, "model.pt")
        torch.save(self.model, model_path)

    def save_loss(self):
        """Save train/val loss as json file to `self.model_dir` directory"""
        loss_path = os.path.join(self.model_dir, "loss.json")
        with open(loss_path, "w") as fp:
            json.dump(self.loss, fp)