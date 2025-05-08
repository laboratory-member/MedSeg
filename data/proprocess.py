import os
import nibabel as nib
import numpy as np

# 路径配置
input_dir = 'ct'
output_dir = 'preprocess'

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

def convert_to_hu(data, slope=1.0, intercept=-1024.0):
    return data * slope + intercept

def normalize(data):
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / std

def preprocess_file(file_path, output_path):
    img = nib.load(file_path)
    data = img.get_fdata()
    header = img.header

    # 获取slope和intercept，使用默认值（slope=1.0, intercept=-1024.0）
    slope = float(header.get('scl_slope', 1.0))
    intercept = float(header.get('scl_inter', -1024.0))

    hu_data = convert_to_hu(data, slope, intercept)
    norm_data = normalize(hu_data)

    norm_img = nib.Nifti1Image(norm_data, img.affine, header)
    nib.save(norm_img, output_path)

if __name__ == '__main__':
    for i in range(1, 108):
        filename = f'{i}.nii.gz'
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        if os.path.exists(input_path):
            print(f'Processing {filename}...')
            preprocess_file(input_path, output_path)
        else:
            print(f'Warning: {filename} not found in {input_dir}.')
