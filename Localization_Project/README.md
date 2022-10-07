
# Localization using ICP and NDT
This is the project for the third course in the  [Udacity Self-Driving Car Engineer Nanodegree Program](https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013) : Localization.

## Project overview
In this project, the aim is to localize the position of a vehicle in Carla simulator using only the data from the 3D LiDAR sensor. I used the Iterative Closest Point (ICP) and Normal Distribution Transform (NDT) scan matching algorithm to solve the localization problem.

## Requirements
To execute this code you should have the following modules:

- Cmake (https://cmake.org/install/)
- Make (https://askubuntu.com/questions/161104/how-do-i-install-make)
- CARLA Simulator (https://carla.org/)
- PCL (Point Cloud Library) (https://pointclouds.org/downloads/)
- Eigen Library for C++ (https://eigen.tuxfamily.org/index.php?title=Main_Page)

## Compile and Run
First you should compile the code using the following commands:
```
cmake .
make
```


After you should run the CARLA simulation `./run_carla.sh`.

After you should run the localization executable `./cloud_loc`

### How to use
To control the vehicle you have the following key:
- Right arrow: Increase wheel angle to the right
- Left arrow: Increase wheel angle to the left
- Up arrow: Increase Speed
- Down arrow: Reduce Speed
- a: Refresh view
- i: Chose ICP matching algorithm
- n: Chose NCT matching algorithm


### Results
In the following GIFs I made the localization of the vehicle using the speed of the vehicle equivalent to press the  Up arrow 3 times and I also change the wheel angle to generate some curves in the path.

The point cloud of the map is illustrated by the blue points, and the point cloud of the LIDAR sensor of the vehicle are the red one. The true pose of the vehicle is the red rectangle and the localized pose by the algorithm is the green one. The point cloud are in 3D but to better visualize the matching I took a bird eye view.

With both scan matching algorithm I achieved a max pose error lower than 1.2 m in a distance of 170 m. The following gifs are 4 times faster than the original ones.  

ICP
<img src="gif/ICP.gif"/>
NDT
<img src="gif/NDT.gif"/>
