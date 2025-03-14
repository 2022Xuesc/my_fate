import os
import shutil

# 读取保留的目录列表
# with open('keep_dirs.txt', 'r') as f:
#     keep_dirs = {line.strip() for line in f if line.strip()}

keep_dirs = {
    '202410220416373841590',
    '202410220442101237570',
    '202410220819086555390',
    '202411040306089134600',
    '202411040308244428370'
}
# 获取当前目录下的所有子目录
current_dir = os.getcwd()
for item in os.listdir(current_dir):
    item_path = os.path.join(current_dir, item)
    # 检查是否为目录
    if os.path.isdir(item_path):
        if item in keep_dirs:
            print(f"Deleting directory: {item}")
            shutil.rmtree(item_path)  # 删除目录及其内容
