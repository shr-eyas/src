# Setting up RealSense on ROS

### Install basic tools 
```bash
sudo apt update 
sudo apt install -y librealsense2-utils v4l-utils usbutils jq
```

### List the cameras
```bash
rs-enumerate-devices | egrep "Name|Serial Number|USB Type Descriptor|Physical Port|USB Product ID"
```

### Verify USB port and speed 
```bash
lsusb -t
rs-enumerate-devices
```

### Stop all nodes using the cameras
```bash
rosnode kill -a || true
pkill -f realsense || true
```

### Install Intel udev rules and reload
```bash
sudo apt update
sudo apt install -y librealsense2-udev-rules
sudo udevadm control --reload-rules
sudo udevadm trigger
```

Replug both cameras. Then verify non-root access
```bash
realsense-viewer
```

### Launch using a launch file

```xml
<launch>
  <include file="$(find realsense2_camera)/launch/rs_camera.launch">
    <arg name="camera" value="hand_cam"/>
    <arg name="serial_no" value="342522073042"/>
    <arg name="enable_color" value="true"/>
    <arg name="enable_depth" value="true"/>
    <arg name="align_depth" value="true"/>
    <arg name="publish_tf" value="true"/>
    <arg name="initial_reset" value="true"/>
    <arg name="wait_for_device_timeout" value="10.0"/>
  </include>
</launch>
```


### Set up OpenCV for ROS

```bash
sudo apt install ros-noetic-vision-opencv ros-noetic-tf2-ros ros-noetic-franka-msgs
```

### Install h5py to store in hdf5 format
```bash
pip install --user h5py
```

# Hand-eye camera calibration
We will use [Easy Hand-eye](https://github.com/IFL-CAMP/easy_handeye) for extrinsic camera calibration 


### Bringup the robot in gravity compensation
We will manually move the robot and not use MoveIt!
```bash
roslaunch multi_arm_controllers grav_comp.launch robot_ip:=192.168.1.11
```
### Bringup the eye-in-the-hand RealSense camera
```bash
roslaunch data_collection hand_cam.launch
```

### Bringup the AprilTag detector
Do note that the AprilTag is placed within the view of camera. You can visualize using RViz. 

```bash
roslaunch data_collection hand_tags.launch
```

Some [tips](https://github.com/IFL-CAMP/easy_handeye?tab=readme-ov-file#tips-for-accuracy) for accuracy.

### Perform the calibration
Capture samples from around 30 samples by moving the tag all around the screen while keeping it fixed on ground. Once done, click on compute. This will give you the `tf` beteween `robot_effector_frame` and `tracking_base_frame`.
```bash
roslaunch easy_handeye calibrate.launch         
  eye_on_hand:=true   
  freehand_robot_movement:=true   
  start_rviz:=false   
  robot_base_frame:=fr3_link0   
  robot_effector_frame:=fr3_EE   
  tracking_base_frame:=hand_cam_color_optical_frame   
  tracking_marker_frame:=tag_0
```

### Calculate and launch the required `tf`
Now we have `fr3_EE → hand_cam_color_optical_frame` using above.

Run to get  `hand_cam_link → hand_cam_color_optical_frame`
```bash
rosrun tf tf_echo hand_cam_link hand_cam_color_optical_frame
```

We need one parent chain 
`fr3_link0 → … → fr3_EE → hand_cam_link → hand_cam_color_optical_frame`.

Perform the operation an run the printed line. (You can also add it in a launch file)
```python
import numpy as np

# fr3_EE -> hand_cam_color_optical_frame  (x y z qw)
t_eo=[0.06480783303792627,0.035918074180326706,-0.020903611197928853]
q_eo=[-0.0002415063887532385,0.00060180752372474,-0.7079880892845313,0.7062240755833855]  # x y z w

# hand_cam_link -> hand_cam_color_optical_frame
t_lo=[0.0,0.015,0.0]
q_lo=[0.501,-0.500,0.504,-0.496]  # x y z w

def q_to_R(q):
    x,y,z,w = np.array(q)/np.linalg.norm(q)
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1-2*(x*x+z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1-2*(x*x+y*y)],
    ])

def R_to_q(R):
    tr=R.trace()
    if tr>0:
        S=np.sqrt(tr+1.0)*2; w=0.25*S
        x=(R[2,1]-R[1,2])/S; y=(R[0,2]-R[2,0])/S; z=(R[1,0]-R[0,1])/S
    elif R[0,0]>R[1,1] and R[0,0]>R[2,2]:
        S=np.sqrt(1.0+R[0,0]-R[1,1]-R[2,2])*2
        w=(R[2,1]-R[1,2])/S; x=0.25*S; y=(R[0,1]+R[1,0])/S; z=(R[0,2]+R[2,0])/S
    elif R[1,1]>R[2,2]:
        S=np.sqrt(1.0+R[1,1]-R[0,0]-R[2,2])*2
        w=(R[0,2]-R[2,0])/S; x=(R[0,1]+R[1,0])/S; y=0.25*S; z=(R[1,2]+R[2,1])/S
    else:
        S=np.sqrt(1.0+R[2,2]-R[0,0]-R[1,1])*2
        w=(R[1,0]-R[0,1])/S; x=(R[0,2]+R[2,0])/S; y=(R[1,2]+R[2,1])/S; z=0.25*S
    q=np.array([x,y,z,w]); return (q/np.linalg.norm(q)).tolist()

# T_el = T_eo * inv(T_lo)
R_el = q_to_R(q_eo) @ q_to_R(q_lo).T
t_el = np.array(t_eo) - R_el @ np.array(t_lo)
q_el = R_to_q(R_el)

print("fr3_EE -> hand_cam_link")
print("translation:", t_el.tolist())
print("quaternion:", q_el)
print("\nStatic TF:")
print(f"rosrun tf2_ros static_transform_publisher {t_el[0]} {t_el[1]} {t_el[2]} {q_el[0]} {q_el[1]} {q_el[2]} {q_el[3]} fr3_EE hand_cam_link")
```

Note that we did not directly choose `hand_cam_link` instead of `hand_cam_color_optical_frame` since we get the `tf` of AprilTag with respect to the optical frame and not the base link. 

Add it in a launch file which runs with robot bringup
```xml
<launch>
  <node pkg="tf2_ros" type="static_transform_publisher" name="fr3_handeye_tf" args="
    0.06490541 0.02091871 -0.02080638 
    0.000007737 -0.707028769 0.004591718 0.707169878 fr3_EE hand_cam_link"/>
</launch>
```

# Ground camera calibration
We need to find the `tf` between ground camera link and robot base. 

Launch both RealSense cameras
```bash
roslaunch data_collection dual_realsense.launch
```

Launch both RealSense cameras
```bash
roslaunch data_collection dual_realsense.launch
```

Launch the eye-in-hand `tf`
```bash
roslaunch data_collection fr3_handeye_tf.launch
```
Launch the AprilTag detectors
```bash
roslaunch data_collection dual_tags.launch 
```

Now ensure AprilTag is stationary and with view of both cameras.

Test if all required `tfs` are available
```bash
rostopic echo -n1 /hand_tags/tag_detections
```
```bash
rostopic echo -n1 /ground_tags/tag_detections
```
```bash
rosrun tf tf_echo fr3_link0 hand_cam_color_optical_frame
```
```bash
rosrun tf tf_echo ground_cam_color_optical_frame ground_cam_link
```

If yes, run the Python code to give the required `tf`
```bash
python3 data_collection/scripts/test_calibration.py
```