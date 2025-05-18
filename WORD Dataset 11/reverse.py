import os
import re
import nibabel as nib
import numpy as np

def extract_core_name(filename):
    """
    提取如 word_0001 这样的核心文件名。
    若无匹配，则用原文件名去除扩展名。
    """
    match = re.search(r'mapped_word_\d{4}', filename)
    return match.group(0) if match else os.path.splitext(filename)[0]

def flip_z_and_save(input_folder, output_folder, suffix, save_as_gz=True):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith(suffix):
            input_path = os.path.join(input_folder, filename)

            core_name = extract_core_name(filename)
            output_filename = core_name + '.nii.gz' if save_as_gz else core_name + '.nii'
            output_path = os.path.join(output_folder, output_filename)

            # 载入并翻转
            img = nib.load(input_path)
            data = img.get_fdata()
            flipped_data = data[:, :, ::-1]
            flipped_img = nib.Nifti1Image(flipped_data, img.affine, img.header)

            nib.save(flipped_img, output_path)
            print(f"Saved flipped file: {output_path}")

# 示例调用
flip_z_and_save('turnto11labels', 'label_reversed', '.nii.gz', save_as_gz=True)
