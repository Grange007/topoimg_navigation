﻿#!/bin/bash

# Create a new tmux session
session_name="vint_locobot_$(date +%s)"
tmux new-session -d -s $session_name

# Split the window into four panes
tmux selectp -t 0    # select the first (0) pane
tmux splitw -h -p 50
tmux selectp -t 0    # select the first (0) pane
tmux splitw -v -p 50 # split it into two halves
tmux selectp -t 2    # select the new, second (2) pane
tmux splitw -v -p 50 # split it into two halves

# Run the roslaunch command in the first pane
tmux select-pane -t 0
tmux send-keys "source /home/yzc/project/ros_tools/devel/setup.sh" Enter
tmux send-keys "roslaunch vint_locobot.launch" Enter

# Run the navigate.py script with command line args in the second pane
tmux select-pane -t 1
# tmux send-keys "conda activate vint_deployment" Enter
tmux send-keys "conda activate vint_deployment" Enter
tmux send-keys "python3 navigate_visualize.py $1 $2 $3 $4" Enter

# Run the pd_controller.py script in the fourth pane
tmux select-pane -t 2
tmux send-keys "conda activate vint_deployment" Enter
tmux send-keys "bash ./_two_stage_navigation.sh $1 $2 $3 $4" Enter

tmux select-pane -t 3
tmux send-keys "rviz -d visualize.rviz" Enter

# Attach to the tmux session
tmux -2 attach-session -t $session_name
