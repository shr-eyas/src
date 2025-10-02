## Data Collection Project  

### Add robot IP address (only upon start-up)
Settings

Networkk 

Wired

IPv4

Manual

Address 192.168.1.0

Netmask 255.255.255.0

Apply

```bash
sudo ip addr add 192.168.1.0/24 dev enp0s31f6
```

### Enter the workspace
```bash
cd fr3_ws
```

### Source the bash file
```bash
source devel/setup.bash
```

### Bring the robot to start
```bash
roslaunch fr3_controllers move_to_start_controller.launch robot_ip:=192.168.1.11

roslaunch fr3_controllers move_to_start_controller.launch robot_ip:=192.168.1.15
```

### Start the teleoperation 
```bash
roslaunch multi_arm_controllers mirror_teleop.launch
```

### Start the grippers
Specify the object width in `close_width` argument in meters (m).
```bash
roslaunch data_collection gripper.launch close_width:=0.002
```

### Start the cameras
```bash
roslaunch data_collection combined.launch
```

To check if cameras are on
```bash
 rostopic list
```

### Start the data logging script
```bash
roslaunch data_collection session.launch
```
