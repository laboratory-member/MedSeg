import os
import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm

# 文件夹路径
ct_dir = 'dataset/ct'
label_dir = 'dataset/label'

# 标签范围（不含背景0）
label_ids = range(1, 12)

# 保存结果的列表
results = []

# 获取所有 .nii.gz 文件名（以ct文件为基准）
nii_files = [f for f in os.listdir(ct_dir) if f.endswith('.nii.gz')]

# 遍历文件并显示进度
for filename in tqdm(nii_files, desc="Processing files"):
    ct_path = os.path.join(ct_dir, filename)
    label_path = os.path.join(label_dir, filename)

    if not os.path.exists(label_path):
        print(f"跳过无对应标签的文件: {filename}")
        continue

    try:
        ct_img = nib.load(ct_path).get_fdata()
        label_img = np.rint(nib.load(label_path).get_fdata()).astype(np.int32)  # 强制标签为整数
    except Exception as e:
        print(f"读取文件失败 {filename}: {e}")
        continue

    row = {"filename": filename}

    for label in label_ids:
        mask = label_img == label
        if np.any(mask):
            ct_values = ct_img[mask]
            row[f"tag{label}_min"] = np.min(ct_values)
            row[f"tag{label}_max"] = np.max(ct_values)
        else:
            row[f"tag{label}_min"] = np.nan
            row[f"tag{label}_max"] = np.nan

    results.append(row)

# 转为DataFrame并保存为Excel
df = pd.DataFrame(results)
df.to_excel("hu_min_max.xlsx", index=False)
