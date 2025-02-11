# gnome-terminal -- bash -c "roscore"

gnome-terminal -- bash -c "roslaunch realsense2_camera rs_camera.launch align_depth:=true publish_tf:=true"

gnome-terminal -- bash -c "cd /home/yzc/project/galaxea/B1_SDK; sudo bash ./start_connect.sh; source ./install/setup.bash; roslaunch signal_chassis b1.launch"

