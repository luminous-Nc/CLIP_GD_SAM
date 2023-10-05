import os
def delete_files_with_underscore(root_dir):
    # 遍历根文件夹及其子文件夹
    for root, dirs, files in os.walk(root_dir):
        for filename in files:
            file_path = os.path.join(root, filename)

            # 如果文件名中含有下划线 "_"
            if "_" in filename:
                try:
                    # 删除文件
                    os.remove(file_path)
                    print(f"删除文件: {file_path}")
                except Exception as e:
                    print(f"删除文件 {file_path} 时出现错误: {e}")


if __name__ == "__main__":
    folder_path = "./../dataset/food_103/pure_masks"  # 替换为你要操作的文件夹路径
    delete_files_with_underscore(folder_path)
