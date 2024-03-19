
import os  
import glob 
import requests
from io import BytesIO
from PIL import Image

def download_image(url, save_directory):
    try:
        response = requests.get(url)
        response.raise_for_status()  # 检查请求是否成功

        # 从URL中提取文件名
        filename = os.path.join(save_directory, os.path.basename(url))

        # 使用BytesIO将获取的内容转换为二进制数据
        image_data = BytesIO(response.content)

        # 使用PIL库打开图像
        image = Image.open(image_data)

        # 保存图像到指定目录
        image.save(filename)
    
        print(f"Image downloaded and saved to: {filename}")
        return filename
    except requests.exceptions.RequestException as e:
        print(f"Error fetching image: {e}")



def delete_specific_image(file_path):  
    """  
    删除指定路径下的单个图片文件。  
  
    参数:  
        file_path (str): 要删除的图片文件的完整路径  
    """  
    # 检查文件是否存在  
    if os.path.exists(file_path):  
        # 尝试删除文件  
        try:  
            os.remove(file_path)  
            print(f"已删除文件：{file_path}")  
        except OSError as e:  
            # 如果文件无法删除，打印错误消息  
            print(f"无法删除文件 {file_path}. 原因: {e.strerror}")  
    else:  
        # 如果文件不存在，打印错误消息  
        print(f"文件 {file_path} 不存在")  
  
# 使用示例  
# 请将以下路径替换为您要删除的图片文件的实际路径  
        