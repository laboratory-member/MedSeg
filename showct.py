import numpy as np  # 转换格式
import nibabel as nib   # 读取数据
import matplotlib.pyplot as plt # 单张图像展示
from nibabel.viewers import OrthoSlicer3D   # nii.gz展示
import matplotlib
matplotlib.use('TkAgg') # 用于滚动查看nii.gz

# 使用 nibabel库读取图像
image_path = r"1-0.nii.gz"
image_obj = nib.load(image_path)
print(f'文件路径： {image_path}')
print(f'图像类型： {type(image_obj)}')

# 提取numpy数组
image_data = image_obj.get_fdata()
# print(type(image_data))

# 查看图像大小
depth, height, width = image_data.shape
print(f"The image object height: {height}, width:{width}, depth:{depth}")

# 查看图像值范围
print(f'image value range: [{image_data.min()}, {image_data.max()}]')

# 可视化图像
OrthoSlicer3D(image_obj.dataobj).show()

# 查看图像成像信息
print(image_obj.header.keys())

# 图像分辨率信息
pixdim =  image_obj.header['pixdim']
print(f'z轴分辨率： {pixdim[3]}')
print(f'in plane 分辨率： {pixdim[1]} * {pixdim[2]}')

# 依据层厚信息，以及矩阵大小，就可以求出实际的扫描范围。
z_range = pixdim[3] * depth
x_range = pixdim[1] * height
y_range = pixdim[2] * width
print(f'扫描范围：', x_range, y_range, z_range)

# 查看指定层slice图像
maxval = width  # 修改为width（图像宽度）
for i in range(maxval):  # 从0到width逐层查看
    print(f"Plotting x Layer {i} of Image")
    plt.imshow(image_data[:, :, i], cmap='gray')  # 按照列展示切片
    plt.axis('off')  # 关闭网格
    plt.show()
