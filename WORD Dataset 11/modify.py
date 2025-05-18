import os
import nibabel as nib
import numpy as np

label_dir = "./dataset/ct"

for filename in sorted(os.listdir(label_dir)):
    if not filename.endswith(".nii.gz"):
        continue

    file_path = os.path.join(label_dir, filename)

    try:
        nii = nib.load(file_path)
        label_data = nii.get_fdata()

        # 四舍五入后转为 int32
        label_data_int32 = np.rint(label_data).astype(np.int32)

        # 创建新的 Nifti 图像
        new_nii = nib.Nifti1Image(label_data_int32, affine=nii.affine, header=nii.header)
        nib.save(new_nii, file_path)

        print(f"✅ 转换完成: {filename}")

    except Exception as e:
        print(f"❌ 处理失败 {filename}: {e}")
