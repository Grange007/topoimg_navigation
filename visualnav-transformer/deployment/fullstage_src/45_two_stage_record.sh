second_process() {
    echo "Running rosbag record..."
    cd ../topomaps/bags
    echo $1
    # rosbag record /usb_cam/image_raw -o $1 # change topic if necessary
    rosbag record /usb_cam/image_raw -O $1 # change topic if necessary

}

first_process() {
    echo "Start Image Storer..."
    python ./image_storer.py $1
}

# Change the directory to ../topomaps/bags and run the rosbag record command in the third pane
# trap 'second_process $1' SIGINT

# 启动第一个进程
first_process $1

read -r

second_process $1
