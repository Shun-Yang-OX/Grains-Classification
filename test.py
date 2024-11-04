import os

def count_files_in_directory(directory_path):
    try:
        # 使用列表解析统计文件数量
        file_count = len([f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))])
        return file_count
    except FileNotFoundError:
        return "文件夹路径不存在。"
    except Exception as e:
        return f"发生错误：{e}"

# 示例：指定文件夹路径
directory_path = r'/home/shun/Project/Grains-Classification/Data/test/equiax'
print("文件数量:", count_files_in_directory(directory_path))
