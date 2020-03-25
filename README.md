# Computer Vision based Robot Navigation

Vision-based robot navigation has long been a fundamental goal in both robotics and computer vision research. While the problem is largely solved for robots equipped with active range-finding devices, for a variety of reasons, the task still remains challenging for robots equipped only with vision sensors. Vision is an attractive sensor as it helps in the design of economically viable systems with simpler sensor limitations. It facilitates passive sensing of the environment and provides valuable semantic and depth information about the scene that is unavailable to other sensors. However, we do not require all semantic labelling of a particular environment for a robot to move. It requires only the floor part of a environment to navigate its path as we are dealing with ground based robots. Depth information is particularly useful in predicting how much the robot can move in a particular direction. Hence, the marriage between both the vision tasks provides us a free space map which enables the robot to move freely in a given direction. However, it is not always possible for a robot to navigate in a linear path. Hence, we have designed a novel motion control algorithm which enables the robot to naviagte through obstacles in its path. 

### Demo : 

<p align="center">
  <img src="https://github.com/sauradip/vision_based_robot_navigation/blob/master/images/robot_loco.gif">
</p>

This demo is running on a NVIDIA GTX 1080 Ti GPU at around 5-10 fps. The speed is slow due to some opencv operations whi h I am currently trying to implement in GPU tensor and make it realtime ( 25-30 fps ). 

### Model Architecture : 

![Model Architecture](https://github.com/sauradip/vision_based_robot_navigation/blob/master/images/robot_loco.png)

We have implemented our algorithm using [Multitask RefineNet](https://github.com/DrSleep/multi-task-refinenet). The input to this network is a stream of video and we have only considered semantic segmentation and depth estimation outputs trained on NYUD dataset. This output comes out realtime at 25/30 fps. The floor/non-floor segmentation module activates only the floor segmentation and switches off the remaining labels and treat them as background/default class. Hence it becomes a binary segmentation problem. In other channel, we have depth cut module which iteratively cuts the depth of the visible scene upto 2 metres. The reason behind this approach is that it enables the robot to be aware of its immediate surrounding and take decision based on what it views instead of taking a long term decision. The fusion is a basic pixel wise concatenation and finally we obtain the free space map.


### Motion Control 

<p align="center">
  <img src="https://github.com/sauradip/vision_based_robot_navigation/blob/master/images/path_1.png">
</p>

