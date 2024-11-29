import json
import glob
from collections import defaultdict

data_path = glob.glob('/mnt/aigc_cq/private/feishen/datasets/deepfashing/IMAGPose_DeepFashion_pose/*.[jp][pn]g')
save_path = '/mnt/aigc_cq/private/feishen/datasets/deepfashing/IMAGPose_DeepFashion_label.json'

print(len(data_path))
all_img_path = defaultdict(list)
for img_path in data_path:
    cur_label = '_'.join(img_path.split('/')[-1].split('_')[:-2])
    all_img_path[cur_label].append(img_path.replace('IMAGPose_DeepFashion_pose', 'train_all_png'))

# 如果你需要标签列表，可以直接从字典中获取
labels = list(all_img_path.keys())
print(len(labels))
save_list = []
for label in labels:
    cur_list = all_img_path[label]
    save_list.append(cur_list)
with open(save_path, "w") as f:
    f.write(json.dumps(save_list))

# #
# # 以下代码过滤 一个 ID 小于 4的
data = json.load(open(save_path, 'r'))
# 创建一个新的字典来存储长度大于等于4的列表
new_data = []

# 遍历字典并删除长度小于4的列表
for sublist in data:
    if isinstance(sublist, list) and len([f for f in sublist if f.endswith(".png") or f.endswith(".jpg") or f.endswith(".webp") or f.endswith(".jpeg") or f.endswith(".bmp")]) >= 2:  # 确保值是列表且长度大于等于4
        new_data.append(sublist)
print(len(new_data))
# 保存到新的 JSON 文件
with open(save_path, 'w') as f:
    json.dump(new_data, f)

print("Done!")