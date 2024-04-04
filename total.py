import pandas as pd
import matplotlib.pyplot as plt

# 读取Excel文件
df1 = pd.read_excel('elo_1.xlsx')
df2 = pd.read_excel('elo_2.xlsx')

# 假设我们要合并的是第一列，你可以根据需要调整列名或列索引
# 这里也假设给定的比例是0.5，意味着我们想要两个数据集的平均值
ratio = 0.5
player1_elo_pred_ad = df1.iloc[:, 1] * ratio + df2.iloc[:, 1] * (1 - ratio)
player2_elo_pred_ad = df1.iloc[:, 2] * ratio + df2.iloc[:, 2] * (1 - ratio)

plt.plot(player1_elo_pred_ad, label= 'Carlos Alcaraz')
plt.plot(player2_elo_pred_ad, label= 'Jeremy Chardy')
plt.legend()
plt.show()