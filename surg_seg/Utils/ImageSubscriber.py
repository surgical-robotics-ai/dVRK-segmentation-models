import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import numpy as np
import cv2
from pathlib import Path


class ImageSubscriber:
    def __init__(self):
        """This class will not work unless you initialize a ROS topic

        rospy.init_node("image_listener")
        """

        if not rospy.get_node_uri():
            rospy.init_node("image_saver_node", anonymous=True, log_level=rospy.WARN)
        else:
            rospy.logdebug(rospy.get_caller_id() + " -> ROS already initialized")

        self.bridge = CvBridge()
        self.left_cam_subs = rospy.Subscriber(
            "/ambf/env/cameras/cameraL/ImageData", Image, self.left_callback
        )
        self.right_cam_subs = rospy.Subscriber(
            "/ambf/env/cameras/cameraR/ImageData", Image, self.right_callback
        )
        self.left_frame = None
        self.left_ts = None
        self.right_frame = None
        self.right_ts = None

        # Wait a until subscribers and publishers are ready
        rospy.sleep(0.6)

    def get_current_frame(self, camera_selector: str) -> np.ndarray:
        if camera_selector == "left":
            return self.left_frame
        elif camera_selector == "right":
            return self.right_frame
        else:
            raise ValueError("camera selector should be either 'left' or 'right'")

    def left_callback(self, msg):
        try:
            cv2_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.left_frame = cv2_img
            self.left_ts = msg.header.stamp
        except CvBridgeError as e:
            print(e)

    def right_callback(self, msg):
        try:
            cv2_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.right_frame = cv2_img
            self.right_ts = msg.header.stamp
        except CvBridgeError as e:
            print(e)

    def save_frame(self, camera_selector: str, path: Path):
        if camera_selector not in ["left", "right"]:
            ValueError("camera selector error")

        img = self.left_frame if camera_selector == "left" else self.right_frame
        ts = self.left_ts if camera_selector == "left" else self.right_ts
        name = camera_selector + "_frame" + ".jpeg"
        # Save frame
        cv2.imwrite(str(path / name), img)  ## Opencv does not work with pathlib
