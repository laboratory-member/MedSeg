import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
import matplotlib
matplotlib.use('TkAgg')

# 选择要保留的器官编号，例如器官编号为 3
target_label = 1

# 加载图像和标签
image_array = nib.load("0.nii.gz").get_fdata()
label_array = nib.load("1.nii.gz").get_fdata().astype(int)

# 转换为二分类：保留 target_label 为 1，其余为 0
binary_label_array = np.where(label_array == target_label, 1, 0)

# 自定义 0 和 1 的颜色：0 为白色，1 为红色
binary_cmap = ListedColormap([
    [1, 1, 1],  # 0 - 背景白色
    [1, 0, 0],  # 1 - 目标器官红色
])

# 创建窗口
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# 初始显示
slice_index = 0
img1 = ax1.imshow(image_array[:, :, slice_index], cmap='gray')
ax1.set_title("原始图像")
ax1.axis('off')

img2 = ax2.imshow(binary_label_array[:, :, slice_index], cmap=binary_cmap, vmin=0, vmax=1)
ax2.set_title(f"仅显示标签 {target_label}")
ax2.axis('off')

# 更新函数
def update(i):
    img1.set_data(image_array[:, :, i])
    img2.set_data(binary_label_array[:, :, i])
    fig.suptitle(f"Slice: {i}", fontsize=14)
    return img1, img2

# 动画播放
ani = animation.FuncAnimation(fig, update, frames=image_array.shape[2], interval=100, blit=False)

plt.tight_layout()
plt.show()
