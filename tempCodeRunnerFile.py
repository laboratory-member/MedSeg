import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path

# 选择目标器官标签编号
target_label = 1

# 加载原始图像和初步分割结果
image_array = nib.load("4.nii.gz").get_fdata()
label_array = nib.load("4_seg.nii.gz").get_fdata().astype(int)

# 二值化标签，保留目标器官
binary_label_array = np.where(label_array == target_label, 1, 0)

# 初始化显示窗口
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
slice_index = 0

# 显示原始图像和标签
img_ct = ax.imshow(image_array[:, :, slice_index], cmap='gray')
img_label = ax.imshow(binary_label_array[:, :, slice_index], cmap='Reds', alpha=0.4, vmin=0, vmax=1)
ax.set_title(f"标签 {target_label} 叠加在原始图像上")
ax.axis('off')

# 画笔区域（LassoSelector）
lasso = None
current_mask = np.zeros_like(binary_label_array[:, :, slice_index])

# 设置擦除模式的状态
erase_mode = True

# 更新标签区域
def update_mask(verts):
    global current_mask, erase_mode
    path = Path(verts)
    
    # 擦除模式，清除路径区域的标签
    current_mask = np.zeros_like(binary_label_array[:, :, slice_index])  # 重置当前掩码
    for y in range(binary_label_array.shape[0]):
        for x in range(binary_label_array.shape[1]):
            if path.contains_point((x, y)):
                current_mask[y, x] = 0  # 清除标记

    # 更新标签区域
    binary_label_array[:, :, slice_index] = current_mask
    img_label.set_data(binary_label_array[:, :, slice_index])  # 更新显示
    fig.canvas.draw_idle()

# LassoSelector 用于绘制选择区域
def on_select(verts):
    update_mask(verts)

# 创建 LassoSelector
lasso = LassoSelector(ax, on_select)

# 更新显示切片函数
def update_slice(increment):
    global slice_index
    slice_index += increment
    if slice_index < 0:
        slice_index = 0
    elif slice_index >= image_array.shape[2]:
        slice_index = image_array.shape[2] - 1
    
    img_ct.set_data(image_array[:, :, slice_index])
    img_label.set_data(binary_label_array[:, :, slice_index])
    ax.set_title(f"Slice: {slice_index}", fontsize=14)
    fig.canvas.draw_idle()

# 保存调整后的分割结果
def save_adjusted_result():
    nib.save(nib.Nifti1Image(binary_label_array, np.eye(4)), 'adjusted_segmentation.nii.gz')
    print("已保存调整后的分割结果为 'adjusted_segmentation.nii.gz'")

# 按钮事件：保存调整后的结果
from matplotlib.widgets import Button

# 保存按钮
class SaveButton:
    def __init__(self, ax):
        self.ax = ax
        self.button = Button(ax, 'Save Adjustments')
        self.button.on_clicked(self.on_click)

    def on_click(self, event):
        save_adjusted_result()

# 切片更新按钮
class SliceButton:
    def __init__(self, ax, increment):
        self.ax = ax
        self.increment = increment
        self.button = Button(ax, 'Next Slice' if increment > 0 else 'Previous Slice')
        self.button.on_clicked(self.on_click)

    def on_click(self, event):
        update_slice(self.increment)

# 添加按钮
save_button_ax = fig.add_axes([0.7, 0.02, 0.2, 0.05])  # [左，下，宽，高]
save_button = SaveButton(save_button_ax)

previous_button_ax = fig.add_axes([0.1, 0.02, 0.2, 0.05])  # 上一切片
previous_button = SliceButton(previous_button_ax, increment=-1)

next_button_ax = fig.add_axes([0.4, 0.02, 0.2, 0.05])  # 下一切片
next_button = SliceButton(next_button_ax, increment=1)

plt.tight_layout()
plt.show()
