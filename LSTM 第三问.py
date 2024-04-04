import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from captum.attr import IntegratedGradients

number=2

columns_to_remove_per_task = [
    ['match_id', 'swing',],
    ['match_id', 'swing','P1Momentum'],
    ['match_id', 'swing','P1Momentum', 'P2Momentum'],
    ['match_id', 'swing','P1Momentum', 'P2Momentum', 'Speed_KMH']
]

# 数据预处理
def create_sequences(data,  drop_list, sequence_length=number):
    sequences = []
    labels = []
    for match_id in data['match_id'].unique():
        match_data = data[data['match_id'] == match_id]
        for i in range(sequence_length, len(match_data)):
            seq = match_data.iloc[i - sequence_length:i].drop(drop_list,
                                                              axis=1).values
            label = match_data.iloc[i]['swing']
            sequences.append(seq)
            labels.append(label)
    return np.array(sequences), np.array(labels)



class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attention_weight = nn.Parameter(torch.randn(hidden_dim * 2, 1))

    def forward(self, lstm_output):
        # lstm_output shape: (batch, seq_len, hidden_dim * 2)
        attention_scores = torch.matmul(lstm_output, self.attention_weight).squeeze(2)
        attention_weights = F.softmax(attention_scores, dim=1).unsqueeze(2)
        scored_output = lstm_output * attention_weights
        condensed_output = torch.sum(scored_output, dim=1)
        return condensed_output


# 定义包含双向LSTM和注意力机制的模型
class BiLSTMWithAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout_rate=0.1):
        super(BiLSTMWithAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True, dropout=dropout_rate if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout_rate)
        self.attention = Attention(hidden_dim)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        attn_out = self.attention(lstm_out)
        output = self.fc(attn_out)
        return output


# 定义PyTorch数据集
class TennisDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.float), torch.tensor(self.labels[idx], dtype=torch.float)

#
#
# 训练模型
def train_model(model, train_loader, criterion, optimizer, num_epochs=100):
    epoch_losses = []  # 收集每个epoch的平均损失
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0  # 初始化总损失
        for sequences, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()  # 累加损失
        avg_loss = total_loss / len(train_loader)  # 计算平均损失
        epoch_losses.append(avg_loss)  # 收集当前epoch的平均损失
        print(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss}')
    return epoch_losses


for i,element in enumerate(columns_to_remove_per_task):
    # 加载数据
    data = pd.read_csv('2023-wimbledon-points_normalized.csv')

    sequences, labels = create_sequences(data,element)

    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(sequences, labels, test_size=0.2, random_state=42)

    # 创建数据加载器
    train_dataset = TennisDataset(X_train, y_train)
    test_dataset = TennisDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 模型参数
    model = BiLSTMWithAttention(input_dim=sequences.shape[2], hidden_dim=128, num_layers=2, output_dim=1)
    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练模型并收集损失
    train_losses = train_model(model, train_loader, criterion, optimizer)

    if i==0:
        epochs = range(1, len(train_losses) + 1)
        out_data = pd.DataFrame(list(zip(epochs, train_losses)), columns=['Epoch', f'Loss{i+1}'])
    else:
        new_column=pd.DataFrame(train_losses,columns=[f'Loss{i+1}'])
        out_data=pd.concat([out_data,new_column],axis=1)


# print(out_data)
#
# plt.figure(figsize=(10, 6))  # 设置图表大小
# sns.lineplot(data=data, x='Epoch', y='Average MSE Loss', marker='o', dashes=True)  # 使用关键字参数传递数据
#
# plt.title('MSE Loss of Attention-BiLSTM', fontsize=14)  # 添加标题
# plt.xlabel('Epoch', fontsize=12)  # X轴标签
# plt.ylabel('MSE Loss', fontsize=12)  # Y轴标签
# plt.grid(True)  # 显示网格线
# plt.show()
# 将数据从宽格式转换为长格式
long_data = out_data.melt(id_vars='Epoch', value_vars=['Loss1', 'Loss2', 'Loss3', 'Loss4'],
                          var_name='Tournament', value_name='MSE Loss')

# 将变量名映射到更友好的名称
tournament_mapping = {
    'Loss1': '2021-wimbledon',
    'Loss2': '2021-usopen',
    'Loss3': '2021-frenchopen',
    'Loss4': '2021-ausopen'
}
long_data['Tournament'] = long_data['Tournament'].map(tournament_mapping)

# 设置图表大小
plt.figure(figsize=(10, 6))

# 使用Seaborn绘制散点图
sns.lineplot(data=long_data, x='Epoch', y='MSE Loss', hue='Tournament', palette=['blue', 'green', 'red', 'purple'], style='Tournament', markers=True)
# 添加标题和轴标签
plt.title('MSE Loss of Attention-BiLSTM Across Tournaments', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Average MSE Loss', fontsize=12)
# 显示图例
plt.legend(title='Tournament')
# 显示网格线
plt.grid(True)
# 显示图表
plt.show()

# # Step 1: 提取测试集中最后一场比赛的数据
# last_match_id = data['match_id'].unique()[-4]  # 假设最后一场比赛的match_id是列表中的最后一个
# last_match_data = data[data['match_id'] == last_match_id]
#
# # 创建测试数据的序列
# sequences, _ = create_sequences(last_match_data)
#
# # Step 2: 使用模型进行预测
# model.eval()  # 将模型设置为评估模式
# with torch.no_grad():
#     last_match_sequences = torch.tensor(sequences, dtype=torch.float)
#     predictions = model(last_match_sequences).squeeze()
#     predicted_swings = torch.sigmoid(predictions).numpy()  # 将输出转换为概率
#
# # Step 3: 可视化momentum和标注实际swing和预测的swing
# plt.figure(figsize=(12, 6))
# # 绘制momentum
# sns.lineplot(x=range(len(last_match_data)), y=last_match_data['momentum'].values, label='Momentum')
#
# actual_swings = last_match_data['swing'].values
# print(predicted_swings)
# predicted_swings_rounded = [1 if p >= 0.58 else 0 for p in predicted_swings]
#
# #遍历所有的预测点
# for i in range(len(predicted_swings_rounded)):
#     # 如果预测为swing
#     if predicted_swings_rounded[i] == 1:
#         # 如果实际也是swing，则预测正确，标记为绿色
#         if actual_swings[i+number] == 1:
#             plt.scatter(i+number, last_match_data['momentum'].values[i+number], color='green', label='Correct Prediction', s=50, zorder=5)
#         # 如果实际不是swing，则预测错误，标记为红色
#         else:
#             plt.scatter(i+number, last_match_data['momentum'].values[i+number], color='red', label='Incorrect Prediction' , s=50, zorder=5)
#     # 如果实际是swing但没有预测到，也标记为红色
#     elif actual_swings[i+number] == 1:
#         plt.scatter(i+number, last_match_data['momentum'].values[i+number], color='red', s=50, zorder=5)
#
# plt.title(f'Momentum with Actual and Predicted Swings({last_match_id})')
# plt.xlabel('Point Step')
# plt.ylabel('Momentum')
# handles, labels = plt.gca().get_legend_handles_labels()
# by_label = dict(zip(labels, handles))  # 去重，保留顺序
# plt.legend(by_label.values(), by_label.keys())
# plt.show()

# # 初始化Integrated Gradients
# ig = IntegratedGradients(model)
#
# # 将numpy数组转换为torch张量
# X_test_tensor = torch.tensor(X_test, dtype=torch.float)
#
# # 初始化一个数组来累积归因值
# all_attributions = []
#
# # 遍历测试集中的所有样本
# for i in range(X_test_tensor.size(0)):
#     input_tensor = X_test_tensor[i:i+1]  # 获取当前样本，保持批次维度
#     attributions, _ = ig.attribute(input_tensor, target=0, return_convergence_delta=True)
#     all_attributions.append(attributions)
#
# # 将所有归因值堆叠成一个张量
# all_attributions = torch.cat(all_attributions, dim=0)
#
# # 计算每个特征的平均归因值
# mean_attributions = all_attributions.mean(dim=[0, 1]).squeeze().detach().numpy()
#
# print(sum(np.abs(mean_attributions)))
# # 假设我们有一个特征名称列表
# feature_names = data.columns.tolist()[1:][:-1]
#
# # features_to_ignore = ['P1Momentum', 'P2Momentum', 'Speed_KMH','Speed_MPH']
# features_to_ignore=['P1Momentum', 'P2Momentum', 'Speed_KMH']
# mask = np.isin(feature_names, features_to_ignore, invert=True)
#
# # 使用这个布尔索引数组来过滤 mean_attributions 和 feature_names
# mean_attributions = mean_attributions[mask]
# feature_names = np.array(feature_names)[mask]
#
# # 首先，计算每个特征归因值的绝对值
# abs_attributions = np.abs(mean_attributions)
# # 获取绝对归因值排序后的索引
# sorted_indices = np.argsort(abs_attributions)[::-1]  # 降序排序
#
# # 计算要选择的特征数量（前50%）
# num_features_to_select = len(mean_attributions) // 2
# # 选择前50%的特征索引
# selected_indices = sorted_indices[:num_features_to_select]
#
# # 获取对应的归因值和特征名称
# selected_attributions = mean_attributions[selected_indices]
# selected_attributions=[100*i for i in selected_attributions]
# selected_feature_names = np.array(feature_names)[selected_indices]
#
# # 现在可以使用 selected_attributions 和 selected_feature_names 进行可视化
# plt.figure(figsize=(12, 6))
# plt.bar(selected_feature_names, selected_attributions, color='skyblue')
# plt.xticks(rotation=45, ha="right")
# plt.ylabel('Average Attribution(%)')
# plt.title('Main Factors (Top 50%)')
# plt.tight_layout()  # 调整布局以防止标签重叠
# plt.show()
