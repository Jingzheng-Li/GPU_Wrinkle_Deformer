import os

# 设置文件所在目录路径
directory = "/home/jingzheng/Work/Study/CPP_MINE/GPU_Wrinkle_Deformer/Assets"  # 替换成实际目录路径

# 遍历目录中的文件
for filename in os.listdir(directory):
    # 只处理 .obj 文件
    if filename.endswith(".obj"):
        # 如果文件名符合 body 的规则
        if "_body.obj" in filename:
            # 提取数字部分并重命名
            new_filename = "body_" + filename[:4] + ".obj"
            # 构造完整的文件路径
            old_file = os.path.join(directory, filename)
            new_file = os.path.join(directory, new_filename)
            # 重命名文件
            os.rename(old_file, new_file)
            print(f'Renamed: {filename} -> {new_filename}')
        
        # 如果文件名符合 female_tops_004 的规则
        elif "_female_tops_004.obj" in filename:
            # 提取数字部分并重命名
            new_filename = "female_tops_" + filename[:4] + ".obj"
            # 构造完整的文件路径
            old_file = os.path.join(directory, filename)
            new_file = os.path.join(directory, new_filename)
            # 重命名文件
            os.rename(old_file, new_file)
            print(f'Renamed: {filename} -> {new_filename}')
