import os
import nibabel as nib
import numpy as np
import pandas as pd

# 文件夹路径
ct_dir = 'data/ct'
label_dir = 'data/label'

# 标签范围（不含背景0）
label_ids = range(1, 12)

# 保存结果的列表
results = []

for i in range(1, 108):  # 从1到107
    filename = f"{i}.nii.gz"
    ct_path = os.path.join(ct_dir, filename)
    label_path = os.path.join(label_dir, filename)

    if not os.path.exists(ct_path) or not os.path.exists(label_path):
        print(f"跳过不存在的文件: {filename}")
        continue

    try:
        ct_img = nib.load(ct_path).get_fdata()
        label_img = nib.load(label_path).get_fdata()
    except Exception as e:
        print(f"读取文件失败 {filename}: {e}")
        continue

    row = {"filename": filename}

    for label in label_ids:
        mask = label_img == label
        if np.any(mask):
            ct_values = ct_img[mask]
            row[f"tag{label}_p10"] = np.percentile(ct_values, 10)
            row[f"tag{label}_p90"] = np.percentile(ct_values, 90)
        else:
            row[f"tag{label}_p10"] = np.nan
            row[f"tag{label}_p90"] = np.nan

    results.append(row)

# 转为DataFrame并保存为Excel
df = pd.DataFrame(results)
df.to_excel("hu_statistics.xlsx", index=False)
