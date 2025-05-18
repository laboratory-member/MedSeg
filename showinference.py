import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
matplotlib.use('TkAgg')  # 使用交互式后端以支持动画播放

# 选择目标器官标签编号
target_label = 1

# 载入图像和标签
image_array = nib.load("word_0001_0.nii.gz").get_fdata()
label_array = nib.load("word_0001.nii.gz").get_fdata().astype(int)

# 构建二值标签掩码
binary_label_array = np.where(label_array == target_label, 1, 0)

# 创建绘图窗口
fig, ax = plt.subplots(1, 1, figsize=(6, 6))

# 初始化显示第一层切片
slice_index = 0
img_ct = ax.imshow(image_array[:, :, slice_index], cmap='gray')
img_label = ax.imshow(binary_label_array[:, :, slice_index], cmap='Reds', alpha=0.4, vmin=0, vmax=1)

ax.set_title(f"标签 {target_label} 叠加在原始图像上")
ax.axis('off')

# 更新函数（用于动画播放）
def update(i):
    img_ct.set_data(image_array[:, :, i])
    img_label.set_data(binary_label_array[:, :, i])
    ax.set_title(f"Slice: {i}", fontsize=14)
    return img_ct, img_label

# 动画播放
ani = animation.FuncAnimation(fig, update, frames=image_array.shape[2], interval=300, blit=False)

plt.tight_layout()
plt.show()
