import pandas as pd
import os
import numpy as np
import math
# 读取normalized.py中的DataFrame

exp_name = 'aifb'
#normalized = pd.read_csv('aifb/aifb_diff.csv', encoding='utf-8-sig')
normalized = pd.read_csv('./dataset/aifb_centrality_nodes.csv', encoding='utf-8-sig')

origin = pd.DataFrame()

# 遍历path中的文件
path =  "output_new/AIFB/BIDI_output/exp_" + '1'

# 请替换为实际的文件路径
for i in range(1, 10):  # 假设文件名分别为origin-1.csv到origin-10.csv
    print(i)
    filename =  "/Difficult_result_" + '1.5' + "_" + str(i) + ".csv"
    filepath = path + filename

    origin_i = pd.read_csv(filepath, encoding='utf-8-sig')

    origin = pd.concat([origin, origin_i])


def determine_detail(series):
    # 如果0的数量多于非0的数量，则返回0；否则返回1
    return 0 if (series == 0).sum() > (series != 0).sum() else 1


aggregated = origin.groupby('person').agg({
    'sign': 'mean',
    'difficult': 'mean',
    'predict': 'mean',
    'loss': 'mean',
    'detail': determine_detail
}).reset_index()

merged = aggregated

# 检查是否存在重复的person
duplicates = merged[merged.duplicated(subset='person')]

# 打印重复的person
if len(duplicates) > 0:
    print("存在重复的person:")
    print(duplicates['person'])
else:
    print("没有重复的person")
    print(len(merged))
#
merged = merged.merge(normalized[['person', 'difficult']], on='person', how='left')
#

# 存储合并后的结果到指定路径
output_path = 'output-2/aifb/acc951'  # 请替换为实际的输出路径
os.makedirs(output_path, exist_ok=True)  # 确保输出路径存在
output_file = os.path.join(output_path, 'origin.csv')
merged.to_csv(output_file, index=False)

print(f"合并后的DataFrame已成功保存到 {output_file}")

merged.drop('difficult_x' , axis=1 , inplace=True)
merged.rename(columns = {'difficult_y' :'difficult'} , inplace=True)



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

    # 计算CMCost
    cm_cost = (1 - average_difficulty) * error_rate * difficulty_distribution

    if not np.isnan(cm_cost):
        cm_costs = cm_cost + cm_costs

cr = 1-cm_costs
print("cr:",cr)