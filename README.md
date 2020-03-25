# Computer Vision based Robot Navigation

Vision-based robot navigation has long been a fundamental goal in both robotics and computer vision research. While the problem is largely solved for robots equipped with active range-finding devices, for a variety of reasons, the task still remains challenging for robots equipped only with vision sensors. Vision is an attractive sensor as it helps in the design of economically viable systems with simpler sensor limitations. It facilitates passive sensing of the environment and provides valuable semantic and depth information about the scene that is unavailable to other sensors. However, we do not require semantic labelling of a particular environment for a robot to move. It requires only the floor part of a environment to navigate its path as we are dealing with ground based robots. Depth information is particularly useful in predicting how much the robot can move in a particular direction. Hence, the marriage between both the vision tasks provides us a free space map which enables the robot to move freely in a given direction. However, it is not always possible for a robot to navigate in a linear path. Hence, we have designed a novel motion control algorithm which enables the robot to naviagte through obstacles in its path. 

### Demo : 
<p align="center">
  <img src="https://github.com/sauradip/vision_based_robot_navigation/blob/master/images/robot_loco.gif">
</p>

### Model Architecture : 

![Model Architecture](https://github.com/sauradip/vision_based_robot_navigation/blob/master/images/robot_loco.png)

### Motion Control 

<p align="center">
  <img src="https://github.com/sauradip/vision_based_robot_navigation/blob/master/images/path_1.png">
</p>

