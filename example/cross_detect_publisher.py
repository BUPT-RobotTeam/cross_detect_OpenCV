import rospy
import cv2
import numpy as np
import cross_detect
from geometry_msgs.msg import Point2D

rospy.init_node('cross_detect_publisher')
pub = rospy.Publisher('/cross_detect', Point2D, queue_size=10)
rate = rospy.Rate(20)  # 10hz

video = cv2.VideoCapture(1)
img_size = (640, 480)
video.set(cv2.CAP_PROP_FRAME_WIDTH, img_size[0])
video.set(cv2.CAP_PROP_FRAME_HEIGHT, img_size[1])
video.set(cv2.CAP_PROP_FPS, 30)


while not rospy.is_shutdown():
    pnt = Point2D()
    _, frame = video.read()
    pnt.x, pnt.y = cross_detect.cross_detect(frame,True,True,True)
    pub.publish(pnt)
    rate.sleep()
