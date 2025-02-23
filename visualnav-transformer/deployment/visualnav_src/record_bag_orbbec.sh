#!/bin/bash

# Create a new tmux session
session_name="record_bag_$(date +%s)"
tmux new-session -d -s $session_name

# Split the window into three panes
tmux selectp -t 0    # select the first (0) pane
tmux splitw -v -p 50 # split it into two halves
tmux selectp -t 0    # go back to the first pane
tmux splitw -h -p 50 # split it into two halves
tmux selectp -t 2
tmux splitw -h -p 50

# Run the roslaunch command in the first pane
tmux select-pane -t 0
tmux send-keys "source ../../../detic/devel/setup.sh" Enter
tmux send-keys "roslaunch vint_locobot.launch" Enter

# Run the teleop.py script in the second pane
tmux select-pane -t 1
tmux send-keys "source ../../../detic/devel/setup.sh" Enter
tmux send-keys "roslaunch orbbec_camera femto_bolt.launch enable_colored_point_cloud:=true" Enter

# Change the directory to ../topomaps/bags and run the rosbag record command in the third pane
tmux select-pane -t 2
tmux send-keys "cd ../topomaps/bags" Enter
tmux send-keys "rosbag record /camera/color/image_raw -o $1" # change topic if necessary

tmux select-pane -t 3
tmux send-keys "python3 teleop_speed_limit.py" Enter

# Attach to the tmux session
tmux -2 attach-session -t $session_name