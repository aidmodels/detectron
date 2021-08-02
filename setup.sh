apt update
apt install -y git
apt install -y libgtk2.0-dev
apt install -y libjpeg-dev
pip3 uninstall -y Pillow
pip3 install Pillow==6.1
pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
git clone https://github.com/facebookresearch/detectron2 detectron2_repo
pip install -e detectron2_repo
