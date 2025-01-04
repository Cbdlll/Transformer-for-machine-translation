import os
import shutil

def truncate_files(source_dir, target_dir, max_lines=1000):
    """
    Truncate all files in the specified source directory to the first `max_lines` lines,
    and save the truncated files to the target directory.

    Parameters:
        source_dir (str): Path to the source directory containing files to truncate.
        target_dir (str): Path to the target directory to save truncated files.
        max_lines (int): Number of lines to keep in each file.
    """
    # 创建目标目录（如果不存在）
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # 遍历源目录中的文件
    for filename in os.listdir(source_dir):
        source_file_path = os.path.join(source_dir, filename)
        target_file_path = os.path.join(target_dir, filename)

        if os.path.isfile(source_file_path):
            # 读取文件的前 max_lines 行
            with open(source_file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()[:max_lines]

            # 将截断后的内容写入目标文件
            with open(target_file_path, 'w', encoding='utf-8') as file:
                file.writelines(lines)

            print(f"Truncated {source_file_path} to the first {max_lines} lines and saved to {target_file_path}")

if __name__ == "__main__":
    # 定义源目录和目标目录
    source_base_dir = "./data"
    target_base_dir = "./data_1000"

    # 处理 test、train 和 valid 文件夹
    for folder in ["test", "train", "valid"]:
        source_dir = os.path.join(source_base_dir, folder)
        target_dir = os.path.join(target_base_dir, folder)

        if os.path.exists(source_dir):
            truncate_files(source_dir, target_dir)
            print(f"Truncated all files in {source_dir} and saved to {target_dir}")
        else:
            print(f"Source directory {source_dir} does not exist.")
    