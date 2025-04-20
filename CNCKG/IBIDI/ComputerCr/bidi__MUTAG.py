import pandas as pd
import os
import numpy as np
import math
# 读取normalized.py中的DataFrame

exp_name = 'mutag'
# normalized = pd.read_csv('mutag/mutag_diff.csv', encoding='utf-8-sig')
normalized = pd.read_csv('mutag/mutag_centrality_nodes.csv', encoding='utf-8-sig')
# 创建一个空的DataFrame，用于存储合并后的结果
origin = pd.DataFrame()

# 遍历path中的文件
path =  'output_new/MUTAG/BIDI_output/exp_' + '1'

# 请替换为实际的文件路径
for i in range(1, 11):  # 假设文件名分别为origin-1.csv到origin-10.csv
    print(i)
    filename =  "/Difficult_result_" + '9.0' + "_" + str(i) + ".csv"
    filepath = path + filename

    # 读取当前文件的DataFrame
    origin_i = pd.read_csv(filepath, encoding='utf-8-sig')

    # 将当前DataFrame添加到合并结果中
    origin = pd.concat([origin, origin_i])

# 假设origin是您已经通过上述代码合并所有文件后得到的DataFrame

# 定义一个函数来决定detail列的值
def determine_detail(series):
    # 如果0的数量多于非0的数量，则返回0；否则返回1
    return 0 if (series == 0).sum() > (series != 0).sum() else 1

# 使用groupby和agg方法来合并重复的bond数据
# 对于detail列使用自定义的determine_detail函数
# 对于sign, difficult, predict, loss列计算均值
aggregated = origin.groupby('bond').agg({
    'sign': 'mean',
    'difficult': 'mean',
    'predict': 'mean',
    'loss': 'mean',
    'detail': determine_detail
}).reset_index()

merged = aggregated

# 检查是否存在重复的bond
duplicates = merged[merged.duplicated(subset='bond')]

# 打印重复的bond
if len(duplicates) > 0:
    print("存在重复的bond:")
    print(duplicates['bond'])
else:
    print("没有重复的bond")
    print(len(merged))
#
merged = merged.merge(normalized[['bond', 'difficult']], on='bond', how='left')
#

# 存储合并后的结果到指定路径
output_path = 'output-2/aifb/acc951'  # 请替换为实际的输出路径
os.makedirs(output_path, exist_ok=True)  # 确保输出路径存在
output_file = os.path.join(output_path, 'origin.csv')
merged.to_csv(output_file, index=False)

print(f"合并后的DataFrame已成功保存到 {output_file}")

merged.drop('difficult_x' , axis=1 , inplace=True)
merged.rename(columns = {'difficult_y' :'difficult'} , inplace=True)
#
# # 这样merged就可以将difficult均分了
# merged['difficult'] = pd.to_numeric(merged['difficult'], errors='coerce')
#
# # 从大到小排序
# merged = merged.sort_values('difficult', ascending=True)
#
# group_dict ={'aifb':20}
# group_num = group_dict[exp_name]
#
# # 使用 qcut 分组，这里我们直接用分组结果作为新列
# merged['group'] = pd.qcut(merged['difficult'], group_num, labels=range(1, group_num+1))
#
# # 初始化一个空字典来存储每个分组的DataFrame
# group_dfs = {}
#
# # 按组号分组，然后遍历每个组
# for group_name, group_df in merged.groupby('group'):
#     # 将每个组的数据存储为单独的DataFrame
#     group_dfs[group_name] = group_df
#
# # 现在 group_dfs 字典包含了 10 个分组的DataFrame，可以通过组号访问
# # 例如，访问第1组的DataFrame：
# print(group_dfs[1])


merged = merged.sort_values('difficult', ascending=True)

# 定义10个等间隔的bins
bins = [i/10.0 for i in range(11)]  # 生成0, 0.1, 0.2, ..., 1.0 的列表

# 使用cut进行分组
merged['group'] = pd.cut(merged['difficult'], bins=bins, include_lowest=True, right=True)

# 初始化一个空字典来存储每个分组的DataFrame
group_dfs = {}

i=1
# 按组号分组，然后遍历每个组
for group_name, group_df in merged.groupby('group'):
    # 将每个组的数据存储为单独的DataFrame
    group_dfs[i] = group_df
    i = i+1

print(group_dfs[1])

# 首先计算每个组的平均难度 （dict）
average_difficulties = {group: df['difficult'].mean() for group, df in group_dfs.items()}

count_zero = (merged['detail']==0).sum()

# 然后，对于每个组，计算CMCost
cm_costs = 0
for group, df in group_dfs.items():
    average_difficulty = average_difficulties[group]

    error_rate = (df['detail']==0).sum() / count_zero # 这应该是当前组的错误率分布
    difficulty_distribution = len(df)/len(merged)  # 这应该是当前组的难度分布数据
    print(f'{group}组的错误率为{error_rate},平均难度为{average_difficulty}')

    # 计算CMCost
    cm_cost = (1 - average_difficulty) * error_rate * difficulty_distribution

    if not np.isnan(cm_cost):
        cm_costs = cm_cost + cm_costs       # 或者其他适当的聚合方式，取决于您的需求

cr = 1-cm_costs
print("cr:",cr)