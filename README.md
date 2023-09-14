# Cross_Detect_OpenCV

基于OpenCV的白色十字识别，用于农业机器人的自动导航。

## 使用说明

安装requirements.txt中的依赖包

```bash
pip install -r requirements.txt
```

运行main.py

```bash
python cross_detect.py
```

## 项目结构

+ cross_detect.py
  + 主程序，包含十字识别的主要逻辑
+ capture.py
  + 用于拍摄图片，便于调试
+ undistort.py
  + 载入相机参数，对图片进行去畸变
  + 应注意到本项目使用的相机是S908相机，因此需要根据自己的相机参数进行修改
