import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
import matplotlib
matplotlib.use('TkAgg')

# 加载图像和标签
image_array = nib.load("word_0001_0.nii.gz").get_fdata()
label_array = nib.load("word_0001.nii.gz").get_fdata().astype(int)

print(f"Image shape: {image_array.shape}, Label shape: {label_array.shape}")
print(f"Label unique values: {np.unique(label_array)}")

# 自定义标签颜色（最多12种标签）
label_colors = [
    [1, 1, 1],      # 0 - 白色
    [1, 0, 0],      # 1 - 红
    [0, 1, 0],      # 2 - 绿
    [0, 0, 1],      # 3 - 蓝
    [1, 1, 0],      # 4 - 黄
    [1, 0, 1],      # 5 - 紫
    [0, 1, 1],      # 6 - 青
    [0.5, 0.5, 0.5],# 7 - 灰
    [1, 0.5, 0],    # 8 - 橙
    [0.5, 0, 1],    # 9 - 紫
    [0, 0.5, 1],    # 10 - 天蓝
    [0.5, 1, 0.5]   # 11 - 青绿
]
custom_cmap = ListedColormap(label_colors[:int(np.max(label_array)) + 1])

# 创建窗口
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# 初始显示
slice_index = 0
img1 = ax1.imshow(image_array[:, :, slice_index], cmap='gray')
ax1.set_title("原始图像")
ax1.axis('off')

img2 = ax2.imshow(label_array[:, :, slice_index], cmap=custom_cmap, vmin=0, vmax=len(label_colors) - 1)
ax2.set_title("标签图像")
ax2.axis('off')

# 更新函数
def update(i):
    img1.set_data(image_array[:, :, i])
    img2.set_data(label_array[:, :, i])
    fig.suptitle(f"Slice: {i}", fontsize=14)
    return img1, img2

# 动画播放
ani = animation.FuncAnimation(fig, update, frames=image_array.shape[2], interval=100, blit=False)

plt.tight_layout()
plt.show()
