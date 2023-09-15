# Cross_Detect_OpenCV

基于OpenCV的白色十字识别，用于农业机器人的自动导航。

## 使用说明

安装requirements.txt中的依赖包

```bash
pip install -r requirements.txt
```

运行cross_detect.py

```bash
python cross_detect.py
```

你将会需要选择照片模式或者相机模式，照片模式将会从文件夹中读取图片，相机模式将会从相机中读取图片。

如果你选择了相机模式，你需要输入相机的编号，一般来说，编号为0的相机是笔记本自带的摄像头，编号为1的相机是外接的USB摄像头。

如果你选择了照片模式，你需要输入图片的路径，图片的路径应该是相对于cross_detect.py的路径。


## 二次开发

### 项目结构

+ cross_detect.py
  + 十字识别的主要逻辑
+ capture.py
  + 用于拍摄图片，便于调试
+ undistort.py
  + 载入相机参数，对图片进行去畸变
  + 应注意到本项目使用的相机是S908相机，因此需要根据自己的相机参数进行修改

### 调用说明

如果你需要识别十字，你只需要调用cross_detect.py中的cross_detect函数即可

传入参数如下:
+ img: 传入的图片
+ show_source: 是否显示原图，默认为False
+ show_edge: 是否显示边缘检测后的图片，默认为False
+ need_undistort: 是否需要去畸变，默认为False

该函数将返回十字中心的坐标，如果没有检测到十字，将返回None

注意，去畸变需要载入相机参数，你需要修改undistort.py中的相机参数，或者自己写一个去畸变的函数。

相机标定方法可以使用ROS_Camera_Calibration包，也可以使用OpenCV自带的标定方法。

本代码使用的hsv阈值为

```python
hsv_low = np.array([108, 36, 141])
hsv_high = np.array([121, 163, 255])
```

十字的宽度阈值为

```python
cross_width = 50
```

可以根据自己的需求进行修改。





