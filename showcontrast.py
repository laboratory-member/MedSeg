import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
import matplotlib
matplotlib.use('TkAgg')

# 加载图像和标签
image_array = nib.load("0.nii.gz").get_fdata()
label1_array = nib.load("1.nii.gz").get_fdata().astype(int)
label2_array = nib.load("1_seg.nii.gz").get_fdata().astype(int)

# 保留目标标签编号（例如1），转换为二值图
target_label = 1
binary_label1 = np.where(label1_array == target_label, 1, 0)
binary_label2 = np.where(label2_array == target_label, 1, 0)

# 自定义颜色映射（背景白，目标红）
binary_cmap = ListedColormap([
    [1, 1, 1],  # 背景
    [1, 0, 0],  # 目标标签
])

# 创建窗口，三列显示
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

# 初始显示
slice_index = 0
img1 = ax1.imshow(image_array[:, :, slice_index], cmap='gray')
ax1.set_title("原始图像")
ax1.axis('off')

img2 = ax2.imshow(binary_label1[:, :, slice_index], cmap=binary_cmap, vmin=0, vmax=1)
ax2.set_title("标签 1.nii.gz")
ax2.axis('off')

img3 = ax3.imshow(binary_label2[:, :, slice_index], cmap=binary_cmap, vmin=0, vmax=1)
ax3.set_title("标签 1_seg.nii.gz")
ax3.axis('off')

# 更新函数
def update(i):
    img1.set_data(image_array[:, :, i])
    img2.set_data(binary_label1[:, :, i])
    img3.set_data(binary_label2[:, :, i])
    fig.suptitle(f"Slice: {i}", fontsize=14)
    return img1, img2, img3

# 动画播放
ani = animation.FuncAnimation(fig, update, frames=image_array.shape[2], interval=100, blit=False)

# 保存为 MP4
ani.save("output.mp4", writer='ffmpeg', fps=10)

# 如果还想显示窗口，也可以保留
plt.tight_layout()
plt.show()
