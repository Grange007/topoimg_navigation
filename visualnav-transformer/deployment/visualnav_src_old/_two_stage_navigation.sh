python3 ./pd_controller.py

echo "First process terminated, starting second process..."
gnome-terminal -- bash -c "roslaunch pure_odom.launch"

sleep 5

gnome-terminal -- bash -c "source ~/.bashrc; export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtiff.so.5; conda activate vint_deployment; python3 second_navigation.py $1 $2 $3 $4"

sleep 5

gnome-terminal -- bash -c "source ~/.bashrc; conda activate vint_deployment; python3 pid.py"
