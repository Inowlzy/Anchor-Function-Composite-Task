import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from transformer import Encoder
from tqdm import tqdm
from sequence_gen import seq_gen_train, seq_gen_test_data, seq_gen_test_task
import torch.optim.lr_scheduler as lr_scheduler


# 训练和测试函数
def train(model, dataloader, criterion, optimizer, device, writer, epoch):
    model.train()
    total_loss = 0
    correct = 0
    for i, (seq, target, mask, src_mask) in enumerate(dataloader):
        seq, target, mask, src_mask = seq.to(device), target.to(device), mask.to(device), src_mask.to(device)
        optimizer.zero_grad()
        src_mask_t = src_mask.unsqueeze(1).unsqueeze(2)

        output = model(seq, src_mask_t)

        mask_expanded = mask.unsqueeze(-1)
        output_mask = output * mask_expanded.to(device)
        target_mask = target * mask_expanded.to(device)

        loss = criterion(output_mask, target_mask)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        preds = torch.argmax(output_mask, dim=-1)

        target_max = torch.argmax(target, dim=-1)
        temp = (preds == target_max) * mask.to(device)
        correct += (temp).sum().item()

        if i % 10 == 0:
            writer.add_scalar('Train/Loss', loss.item(), epoch * len(dataloader) + i)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / (len(dataloader.dataset))
    writer.add_scalar('Train/Accuracy', accuracy, epoch)

    return avg_loss, accuracy


def test_data(model, dataloader, criterion, device, writer, epoch):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for seq, target, mask, src_mask in dataloader:
            seq, target, mask, src_mask = seq.to(device), target.to(device), mask, src_mask.to(device)
            src_mask_t = src_mask.unsqueeze(1).unsqueeze(2)
            # src_mask = None
            output = model(seq, src_mask_t)
            mask_expanded = mask.unsqueeze(-1)

            # output_temp = output 
            output_mask = output * mask_expanded.to(device)
            target_mask = target * mask_expanded.to(device)

            loss = criterion(output_mask, target_mask)
            total_loss += loss.item()

            preds = torch.argmax(output_mask, dim=-1)

            target_max = torch.argmax(target_mask, dim=-1)
            temp = (preds == target_max) * mask.to(device)
            correct += (temp).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / (len(dataloader.dataset))
    writer.add_scalar('Test Data/Loss', avg_loss, epoch)
    writer.add_scalar('Test Data/Accuracy', accuracy, epoch)

    return avg_loss, accuracy


def test_task(model, dataloader, criterion, device, writer, epoch):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for seq, target, mask, src_mask in dataloader:
            seq, target, mask, src_mask = seq.to(device), target.to(device), mask, src_mask.to(device)
            src_mask_t = src_mask.unsqueeze(1).unsqueeze(2)
            # src_mask = None
            output = model(seq, src_mask_t)
            mask_expanded = mask.unsqueeze(-1)

            # output_temp = output
            output_mask = output * mask_expanded.to(device)
            target_mask = target * mask_expanded.to(device)

            loss = criterion(output_mask, target_mask)
            total_loss += loss.item()

            preds = torch.argmax(output_mask, dim=-1)
            target_max = torch.argmax(target_mask, dim=-1)

            temp = (preds == target_max) * mask.to(device)
            correct += (temp).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / (len(dataloader.dataset))
    writer.add_scalar('Test Task/Loss', avg_loss, epoch)
    writer.add_scalar('Test Task/Accuracy', accuracy, epoch)

    return avg_loss, accuracy


def one_hot_encode(sequence, num_classes):
    one_hot_encoded = np.zeros((len(sequence), num_classes))
    for idx, val in enumerate(sequence):
        one_hot_encoded[idx, val - 1] = 1
    return one_hot_encoded


# 自定义数据集类
class TrainDataset(Dataset):
    def __init__(self, num_samples, num_classes=117):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.sequences = self._generate_sequence(num_samples)
        # print(self.sequences)

    def _generate_sequence(self, num_samples):
        seq = seq_gen_train(num_samples)
        return seq

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        seq, seq_one_hot, masks, src_masks = self.sequences[idx]
        return torch.tensor(seq, dtype=torch.long), torch.tensor(seq_one_hot, dtype=torch.float), torch.tensor(masks,
                                                                                                               dtype=torch.float), torch.tensor(
            src_masks, dtype=torch.float)


class TestDataset(Dataset):
    def __init__(self, num_samples, ty, num_classes=117):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.ty = ty
        self.sequences = self._generate_sequence(num_samples)

    def _generate_sequence(self, num_samples):
        if self.ty == "data":
            seq = seq_gen_test_data(num_samples)
        elif self.ty == "task":
            seq = seq_gen_test_task(num_samples)
        return seq

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        seq, seq_one_hot, masks, src_masks = self.sequences[idx]
        return torch.tensor(seq, dtype=torch.long), torch.tensor(seq_one_hot, dtype=torch.float), torch.tensor(masks,
                                                                                                               dtype=torch.float), torch.tensor(
            src_masks, dtype=torch.float)

import math
class CustomLRScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, lr_start=2e-5, lr_end=2e-4):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.lr_start = lr_start
        self.lr_end = lr_end
        self.current_epoch = 0

    def get_lr(self, epoch):
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.lr_start + epoch * (self.lr_end - self.lr_start) / self.warmup_epochs
        else:
            # Cosine annealing
            t = epoch - self.warmup_epochs
            T_total = self.total_epochs - self.warmup_epochs
            lr = self.lr_end + 0.5 * (self.lr_start - self.lr_end) * (1 + math.cos(t * math.pi / T_total))
        return lr

    def step(self):
        lr = self.get_lr(self.current_epoch)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.current_epoch += 1


# 主函数
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc_voc_size = 10000
    max_len = 9
    d_model = 512
    ffn_hidden = 128
    n_head = 4
    n_layers = 4
    drop_prob = 0.3
    num_epochs = 4000
    batch_size = 100
    learning_rate = 2e-5

    model = Encoder(enc_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device).to(device)
    model.load_state_dict(torch.load("best_model11.pth"))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = CustomLRScheduler(optimizer, warmup_epochs=400, total_epochs=4000, lr_start=2e-5, lr_end=2e-4)

    train_dataset = TrainDataset(num_samples=7000)
    test_data_dataset = TestDataset(1000, "data")
    test_task_dataset = TestDataset(1000, "task")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(test_data_dataset, batch_size=batch_size, shuffle=False)
    test_task_loader = DataLoader(test_task_dataset, batch_size=batch_size, shuffle=False)

    writer = SummaryWriter()
    best_acc = 0

    # for train and test meanwhile
    for epoch in tqdm(range(num_epochs)):
        print('learning rate: ', optimizer.param_groups[0]['lr'])
        train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, device, writer, epoch)
        test_data_loss, test_data_accuracy = test_data(model, test_data_loader, criterion, device, writer, epoch)
        test_task_loss, test_task_accuracy = test_task(model, test_task_loader, criterion, device, writer, epoch)

        scheduler.step()

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f},"
              f"\n Test Data Loss: {test_data_loss:.4f}, Test Data Accuracy: {test_data_accuracy:.4f},"
              f"\n Test Task Loss: {test_task_loss:.4f}, Test Task Accuracy: {test_task_accuracy:.4f}")

        if test_data_accuracy > best_acc:
            best_acc = test_data_accuracy
            print(f"Saving best model with Test Data Accuracy: {best_acc:.4f} "
                  f"and Test Task Accuracy: {test_task_accuracy:.4f}")
            torch.save(model.state_dict(), "best_model11.pth")

    # for test task
    # test_task_loss, test_task_accuracy = test_task(model, test_task_loader, criterion, device, writer, 0)

    writer.close()


if __name__ == "__main__":
    main()
