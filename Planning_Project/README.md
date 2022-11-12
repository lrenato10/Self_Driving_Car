# Motion Planning and Decision Making for Autonomous Vehicles
This is the project for the fourth course in the  [Udacity Self-Driving Car Engineer Nanodegree Program](https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013) : Planning.

## Project Overview
In this project, I made some modification in a code from following repository (https://github.com/udacity/nd013-c5-planning-starter.git). I modified two of the main components of a traditional hierarchical planner: The Behavior Planner and the Motion Planner. Both will work in unison to be able to:

* Avoid static objects (cars, bicycles and trucks) parked on the side of the road (but still invading the lane). The vehicle must avoid crashing with these vehicles by executing either a “nudge” or a “lane change” maneuver.
* Handle any type of intersection (3-way, 4-way intersections and roundabouts) by STOPPING in all of them (by default)
* Track the centerline on the traveling lane.

To accomplish this, I implemented:

* Behavioral planning logic using Finite State Machines - FSM
* Static objects collision checking.
* Path and trajectory generation using cubic spirals
* Best trajectory selection though a cost function evaluation. This cost function will mainly perform a collision check and a proximity check to bring cost higher as we get closer or collide with objects but maintaining a bias to stay closer to the lane center line.

## Compile and Run
* Run Carla Simulator
```
cd /opt/carla-simulator/
SDL_VIDEODRIVER=offscreen ./CarlaUE4.sh -opengl
```
* Configure Machine
```
cd Planning_Project/project
./install-ubuntu.sh
```
* Compile
```
cd Planning_Project/project/starter_files
cmake .
make
``` 
* Run code
```
cd Planning_Project/project
./run_main.sh
```
* If error bind is already in use
  * `ps -aux | grep carla`
  * `kill id`    


# Results
The following accelerated video show the result of the planner (Behavior Planner + the Motion Planner). The input is only the goal point and the planner parameters. The planner will find the trajectory to arrive to this point.  
<img src="planning.gif"/>
