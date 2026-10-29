# Rock Climbing Robots - Whackathon

## Links

Project: <https://rock-climbing-robots.github.io/>

Better Actuator Model: <https://github.com/Rhoban/bam>

Mujoco Models: <https://mujoco.readthedocs.io/en/stable/models.html>

Simulator: <https://github.com/rock-climbing-robots/simulator>

Deploy & Test the robot: <https://github.com/Rhoban/microban>, use branch `dev_2`

Contact: <rock-blimbing-robots@diff.u-bordeaux.fr>

Discord: <http://discord.gg/uhJsQwQcX>

Technical Slides: <https://rock-climbing-robots.github.io/slides/01_toolkit.html?font=sans>

## ToDos

- [ ] Use uv

- [x] What's the team goal? learning? winning? Should we calculate how we could get the most points?
  - Apprendre

## Setup

Quick start
Climber Microban (be careful it's dev_v2 branch)
<https://github.com/Rhoban/microban/tree/dev_v2>

Example repository for RL with Microban
<https://github.com/Rhoban/mjlab_microban/>

Quickstart:

```sh
git clone -b dev_v2 git@github.com:Rhoban/microban.git
cd microban
make sim WALL=N2
make dashboard # And select 127.0.0.1
```

Simulator (used by Microban in make sim)
<https://github.com/rock-climbing-robots/simulator>

Can be used to design wall with:
uv run climbing-sim-wall-editor

## Robots' WiFi

- Robot_Climbing
- SUMMIT2026
(not connected to internet yet)
- each robot will have its own IP
- XXXX = 4 first alphamnumerics of the MAC
- radxa (also has a npu inside, is not a raspi)
- user: radxa, pw: radxa

## Markers

- optional (e.g. Aruco markers)
  - getting extrinsic markers for free
  - pose of camera can be calculated
  - 1 marker on the robot
    - how far the arms can stretch from the marker

- other option: make perception markerless
  - don't hardcode extrinsic

## Challenge

Sufficient: one hand touching the top row (not both hands necessary)

Planning

- inverse kinematics
- reinforcement learning
- hybrid system
- some geometric asusmption & search tree (depending on joints)

## Tips

- use uv
- know your data
- know your robot
- read the docs
- start with the simulator
- (maybe use center of mass)
- DO hold the robot and press `torque off` if the robot moves go array
  - have one student help with the secure
- DON'T print too much / every ms
- DON'T deploy with the robot on the wall, because it will call `torque off` and the robot will fall
- DON'T turn off without using `shutdown` before, that would burn the SD card
- DO `shutdown` before cutting the power
- (wait for the newest version of the robot before doing CAD or CAO)

## Track Levels

3 Different levels

## Setup around wall

- 2 Intel Realsense cameras with 2 usv => plug your own pc
- RGB (depth possible, not necessarily helpful, RGBD)

## Evaluation

Penalties like

1. Height > Time
2. Time

- can be slow to get to the wall if you start on the ground
- 3 Seconds on the hold at the top would be good
Time

## Simulation

- MuJoCo
- rock-climb-robots simulator
  - Angles in radians
  - measures
  - only on simulator (not on robot):
    - e.g. ball positions = center of the ball (25 mm?)
  - Dashboard
    - sliders: e.g move head slider will turn the head of the robot
    - reboot ?
    - emergency off button (`torque off`), hold it first
    - even if torque is off, if you move the robot in real life you will see the 3d model move
    - DON'T deploy with the robot on the wall, because it will call `torque off` and the robot will fall
    - DON'T turn off without using `shutdown` before, that would burn the SD card
    - DO `shutdown` before cutting the power
  - Most robotic people start with :
    - Creating one action that executes motions on *fixed* motions
    - then switch to *parametrized* motionss

## Holes

- only one types
- might have different colors
- orange, green, blue

## Tools

Robot & CAD (wait for the newest version of the robot before doing CAD or CAO)

- Microban
- Onshape
- onshape-to-robot

Modelling

- Pinocchio
- MuJoCo

Inverse kinematics

- pink
- placo
- mink
Specify task atn high level and get it solved automatically (e.g. where you want your hand to be placed)

Planning

- Dijkstra
- A*
- RRT
- RRT-Connect

Reinforcement Learning

- Gymnasium
- Stable Baselines3
  - not optimized for whole body dynamics
  - better for more highlevel problems

multiple mujoco implementatoins

- normal pip install (CPU)
- different version for GPU
- mujoco as backend in mjlab

- there is a mjlab microban repository

- learning the walk takes long (1h on a 4090)

## Graph Search

- Nodes on the example are not the holds, it is the robot's configuration
- maybe not trivial
