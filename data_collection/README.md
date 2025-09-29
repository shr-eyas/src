### Install basic tools for RealSense on ROS
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

### Replug both cameras. Then verify non-root access
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


## Setting up OpenCV with RealSense on ROS

```bash
sudo apt install ros-noetic-vision-opencv ros-noetic-tf2-ros ros-noetic-franka-msgs
```

### Install h5py to store in hdf5 format
```bash
pip install --user h5py
```