import os
import nibabel as nib
import numpy as np

# 输入标签目录
label_dir = './dataset/label'

# 输出的11个器官的子目录：label1 到 label11
output_base = './dataset'
organ_labels = range(1, 12)  # 1 到 11

# 创建输出目录
for organ in organ_labels:
    organ_folder = os.path.join(output_base, f'label{organ}')
    os.makedirs(organ_folder, exist_ok=True)

# 遍历所有 .nii.gz 文件
for filename in os.listdir(label_dir):
    if not filename.endswith('.nii.gz'):
        continue

    filepath = os.path.join(label_dir, filename)

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

print("处理完成：所有 .nii.gz 标签文件已分到 label1 ~ label11 文件夹中。")
