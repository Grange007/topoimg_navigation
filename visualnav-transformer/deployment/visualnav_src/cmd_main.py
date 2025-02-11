import argparse
import subprocess

nav_path1 = "sqz_123_3"
model = "vint"
second_nav_path1 = "sqz_123_1"

import subprocess
import signal
import sys
import os

def robot_launch():
    robot_launch1 = subprocess.Popen([
        "roslaunch",
        "vint_locobot.launch"
    ])
    return robot_launch1

def nav1_launch():
    nav_1 = subprocess.Popen([
        "python",
        "./navigate_visualize.py",
        "--dir",
        nav_path1,
        "--model",
        model
    ])
    return nav_1


def run_command(proc: subprocess.Popen):
    # 启动子进程（创建新进程组）

    def cleanup():
        if proc.poll() is None:  # 检查子进程是否仍在运行
            print("终止子进程...")
            if sys.platform != "win32":
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            else:
                proc.terminate()
            proc.wait()

    # 注册信号处理
    def signal_handler(sig, frame):
        cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # 主程序逻辑...
        proc.wait()
    except Exception as e:
        print(f"发生异常: {e}")
        cleanup()

run_command(robot_launch())
run_command(nav1_launch())

# robot_launch1.terminate()
# nav_1.terminate()
