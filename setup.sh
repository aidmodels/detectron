apt-get update
apt-get install -y git
apt-get install -y libgtk2.0-dev
pip3 uninstall -y Pillow
pip3 install Pillow==6.1
pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
git clone https://github.com/facebookresearch/detectron2 detectron2_repo
pip install -e detectron2_repo
