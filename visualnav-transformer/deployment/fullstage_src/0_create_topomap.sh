#!/bin/bash

traj0_name=$(yq e '.trajs[0]' ../config/traj_name.yaml)
traj1_name=$(yq e '.trajs[1]' ../config/traj_name.yaml)


# Create a new tmux session
session_name="gnm_locobot_$(date +%s)"
tmux new-session -d -s $session_name

# Split the window into three panes
tmux selectp -t 0    # select the first (0) pane
tmux splitw -v -p 50 # split it into two halves
tmux selectp -t 0    # go back to the first pane
tmux splitw -h -p 50 # split it into two halves
tmux selectp -t 2    # go back to the first pane
tmux splitw -h -p 50 # split it into two halves

# Run the create_topoplan.py script with command line args in the second pane
tmux select-pane -t 0
tmux send-keys "conda activate vint_deployment" Enter
tmux send-keys "python3 create_topomap.py --dt 0.06 --dir $traj0_name --topic /image_topic1" Enter

# Run the create_topoplan.py script with command line args in the second pane
tmux select-pane -t 2
tmux send-keys "conda activate vint_deployment" Enter
tmux send-keys "python3 create_topomap.py --dt 0.06 --dir $traj1_name --topic /image_topic2" Enter

sleep 3

# Change the directory to ../topomaps/bags and run the rosbag play command in the third pane
tmux select-pane -t 1
tmux send-keys "mkdir -p ../topomaps/bags" Enter
tmux send-keys "cd ../topomaps/bags" Enter
tmux send-keys "rosbag play -r 9 $traj0_name.bag /usb_cam/image_raw:=/image_topic1" Enter # feel free to change the playback rate to change the edge length in the graph


# Change the directory to ../topomaps/bags and run the rosbag play command in the third pane
tmux select-pane -t 3
tmux send-keys "mkdir -p ../topomaps/bags" Enter
tmux send-keys "cd ../topomaps/bags" Enter
tmux send-keys "rosbag play -r 9 $traj1_name.bag /usb_cam/image_raw:=/image_topic2" Enter

# Attach to the tmux session
tmux -2 attach-session -t $session_name