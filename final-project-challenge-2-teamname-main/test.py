# %%
import os
from sklearn.model_selection import train_test_split
import numpy as np
import shutil
from tqdm import tqdm
# %%
source_root = "/home/ynjuan/final-project-challenge-2-teamname/dataset"
target_root = "/home/ynjuan/final-project-challenge-2-teamname/for-test-dataset"
# %%
from plyfile import PlyData, PlyElement
import pandas as pd
def read_plyfile(filepath):
    with open(filepath, 'rb') as f:
        plydata = PlyData.read(f)
    if plydata.elements:
        return pd.DataFrame(plydata.elements[0].data)
# %%
# 在 test data 中加入 label 這個新的 property
# 只有 test data 需要使用！
os.makedirs(os.path.join(target_root, 'val'), exist_ok=True)
for _, file in enumerate(tqdm(os.listdir(os.path.join(source_root, 'test')))):
    test_path = os.path.join(source_root, 'test', file)
    with open(test_path, 'rb') as f:
        plydata = PlyData.read(f)
    a = read_plyfile(test_path)
    v = plydata.elements[0]
    a = np.empty(len(v.data), v.data.dtype.descr + [('label', 'i4')])
    for name in v.data.dtype.fields:
        a[name] = v[name]
    a['label'] = 0
    # Recreate the PlyElement instance
    v = PlyElement.describe(a, 'vertex')
    # Recreate the PlyData instance
    p = PlyData([v], text=True)
    # pd.DataFrame(p.elements[0].data)
    p.write(os.path.join(target_root, 'val', file))
# %%
data = os.listdir(os.path.join(target_root, 'val'))[0]
print(data)
p = read_plyfile(os.path.join(target_root, 'val', data))
# %%
root = "/home/ynjuan/final-project-challenge-2-teamname/LanguageGroundedSemseg/ckpt/down-stream/Scannet200Voxelization2cmDataset/Res16UNet34D-test-eval/visualize/fulleval"
# 刪除檔案
for file in os.listdir(root):
    if '.ply' in file:
        try:
            os.remove(os.path.join(root, file))
        except OSError as e:
            print(e)
        else:
            print("File is deleted successfully")

# %%
# python read txt file
with open('/home/ynjuan/final-project-challenge-2-teamname/LanguageGroundedSemseg/ckpt/down-stream/Scannet200Voxelization2cmDataset/Res16UNet34D-test-eval/visualize/fulleval/scene0500_00.txt') as f:
    lines = f.readlines()
# %%
# 檢查 download 的 test data 和助教的 test data 有什麼不同
download_path = '/home/ynjuan/final-project-challenge-2-teamname/dataset/train'
downloads = os.listdir(download_path)
my_path = '/home/ynjuan/final-project-challenge-2-teamname/process2/train'
mine = os.listdir(my_path)
for i, file in enumerate(tqdm(mine)):
    d = read_plyfile(os.path.join(download_path, file))
    d = d.drop('label', axis=1)
    d = d.drop('instance_id', axis=1)
    m = read_plyfile(os.path.join(my_path, file))
    m = m.drop('label', axis=1)
    m = m.drop('instance_id', axis=1)
    if m.equals(d):
        pass
    else:
        print('not equal:', file)
    # check x
    if m['x'].tolist() != d['x'].tolist():
        print(f"x is not equal: {file}")
    if m['y'].tolist() != d['y'].tolist():
        print(f"y is not equal: {file}")
    if m['z'].tolist() != d['z'].tolist():
        print(f"z is not equal: {file}")
    if m['red'].tolist() != d['red'].tolist():
        print(f"red is not equal: {file}")
    if m['green'].tolist() != d['green'].tolist():
        print(f"green is not equal: {file}")
    if m['blue'].tolist() != d['blue'].tolist():
        print(f"blue is not equal: {file}")

# %%
# root = '/home/ynjuan/final-project-challenge-2-teamname/dataset'
# train_file_path = "/home/ynjuan/final-project-challenge-2-teamname/dataset/train"
# os.makedirs('/home/ynjuan/final-project-challenge-2-teamname/dataset/train_2', exist_ok=True)
# os.makedirs('/home/ynjuan/final-project-challenge-2-teamname/dataset/val_2', exist_ok=True)
# # %%
# train_file = os.listdir(train_file_path)
# # %%
# train_sample_file,  val_sample_file= train_test_split(train_file, test_size=0.15, random_state=17)
# # %%
# for file in os.listdir(train_file_path):
#     if file in train_sample_file:
#         shutil.copyfile(os.path.join(root, 'train', file), os.path.join(root, 'train_2', file))
#     else:
#         shutil.copyfile(os.path.join(root, 'train', file), os.path.join(root, 'val_2', file))



# # %%
# with open(os.path.join(root, 'train_2.txt'), 'w') as f:
#     for name in train_sample_file:
#         f.write(f'train/{name}')
#         f.write('\n')
# # %%
# with open(os.path.join(root, 'val_2.txt'), 'w') as f:
#     for name in val_sample_file:
#         f.write(f'val/{name}')
#         f.write('\n')
# # %%
# a = os.listdir(os.path.join(root, 'val_2'))
# print(len(a))
# # %%
# a = os.listdir(os.path.join(root, 'train_2'))
# print(len(a))
# %%
