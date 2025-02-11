python3 ./pd_controller.py

echo "First process terminated, starting second process..."
gnome-terminal -- bash -c "roslaunch pure_odom.launch"

# sleep 5

gnome-terminal -- bash -c "source ~/.bashrc; conda activate vint_deployment; python3 second_navigation.py $1 $2 $3 $4"

sleep 7

gnome-terminal -- bash -c "source ~/.bashrc; conda activate vint_deployment; python3 pid.py"
