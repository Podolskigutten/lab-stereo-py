![Ideal stereo geometry](lab-guide/img/ideal_stereo_geometry.png)

# Stereo processing
In this lab we will experiment with stereo processing.

Start by cloning this repository on your machine.
Then, open the project in your editor.

The lab is carried out by following these steps:

1. [Get an overview](lab-guide/1-get-an-overview.md).
2. [Sparse stereo processing](lab-guide/2-sparse-stereo-processing.md).
3. [Dense stereo processing](lab-guide/3-dense-stereo-processing.md).

Please start the lab by going to the [first step](lab-guide/1-get-an-overview.md).

## Prerequisites

Here is a quick reference if you need to set up a Python virtual environment manually:

```bash
python3.8 -m venv venv  # any python version >= 3.8 is OK
source venv/bin/activate.
# expect to see (venv) at the beginning of your prompt.
pip install -U pip  # <-- Important step for Ubuntu 18.04!
pip install -r requirements.txt
```

Please consult the [resource pages] if you need more help with the setup.

[TEK5030]: https://www.uio.no/studier/emner/matnat/its/TEK5030/
[resource pages]: https://tek5030.github.io


## Setup for Jetson (on the lab)
- Clone the repo into the directory `~/tek5030`
- Run the setup script `setup_jetson.bash` which 
  - creates a "venv"
  - installs a lot of stuff
  - downloads a precompiled VTK-wheel
  - download precompiled pyrealsense2
  - installs requirements from `requirements-jetson.txt`
- Open the editor of your choice

```bash
mkdir -p ~/tek5030
cd ~/tek5030
git clone https://github.com/tek5030/lab-stereo-py.git
cd lab-stereo-py
./setup_jetson.bash

# source venv/bin/activate
## connect camera
# python lab_pose_stereo.py
```

:construction: If for some reason things doesn't work, 
- try to run the script `./install_librealsense.bash`
- try to disconnect the camera and reconnect it
- `sudo reboot` the Jetson
