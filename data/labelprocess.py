import os
import nibabel as nib
import numpy as np

# 输入标签目录
label_dir = './data/label'

# 输出的11个器官的子目录：label1 到 label11
output_base = './data'
organ_labels = range(1, 12)  # 1 到 11

# 创建输出目录
for organ in organ_labels:
    organ_folder = os.path.join(output_base, f'label{organ}')
    os.makedirs(organ_folder, exist_ok=True)

# 遍历编号 1 到 107 的标签文件
for i in range(1, 108):
    filename = f"{i}.nii.gz"
    filepath = os.path.join(label_dir, filename)

    if not os.path.exists(filepath):
        print(f"[跳过] 文件不存在: {filepath}")
        continue

    # 加载标签文件
    try:
        img = nib.load(filepath)
        data = img.get_fdata().astype(np.uint8)
        affine = img.affine
        header = img.header
    except Exception as e:
        print(f"[错误] 无法读取 {filepath}：{e}")
        continue

    # 为每个器官创建一个二值掩码
    for organ in organ_labels:
        binary_mask = np.where(data == organ, 1, 0).astype(np.uint8)
        output_img = nib.Nifti1Image(binary_mask, affine, header)
        output_path = os.path.join(output_base, f'label{organ}', filename)
        nib.save(output_img, output_path)

print("处理完成：所有可用标签文件已分到 label1 ~ label11 文件夹中。")
