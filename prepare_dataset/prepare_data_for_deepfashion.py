'Construct a demo of the filtered data set'

import json
import glob
from collections import defaultdict

data_path = glob.glob('./datasets/deepfashing/IMAGPose_DeepFashion_pose/*.[jp][pn]g')
save_path = './datasets/deepfashing/IMAGPose_DeepFashion_label.json'

print(len(data_path))
all_img_path = defaultdict(list)
for img_path in data_path:
    cur_label = '_'.join(img_path.split('/')[-1].split('_')[:-2])
    all_img_path[cur_label].append(img_path.replace('IMAGPose_DeepFashion_pose', 'train_all_png'))


labels = list(all_img_path.keys())
print(len(labels))
save_list = []
for label in labels:
    cur_list = all_img_path[label]
    save_list.append(cur_list)
with open(save_path, "w") as f:
    f.write(json.dumps(save_list))

# #

data = json.load(open(save_path, 'r'))

new_data = []


for sublist in data:
    if isinstance(sublist, list) and len([f for f in sublist if f.endswith(".png") or f.endswith(".jpg") or f.endswith(".webp") or f.endswith(".jpeg") or f.endswith(".bmp")]) >= 2:  # 确保值是列表且长度大于等于4
        new_data.append(sublist)
print(len(new_data))

with open(save_path, 'w') as f:
    json.dump(new_data, f)

print("Done!")