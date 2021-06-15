#!/bin/bash

sudo apt-get update
sudo apt-get install -y git build-essential python3-dev python3-pip python3-venv
if [ ! -f "crawlingathome.py" ]; then
    git clone https://github.com/Wikidepia/crawlingathome-worker
    cd crawlingathome-worker
    python3 -m venv venv && . venv/bin/activate
fi
git clone "https://github.com/TheoCoombes/crawlingathome" crawlingathome_client
pip3 install -r crawlingathome_client/requirements.txt --no-cache-dir
pip3 install -r requirements.txt --no-cache-dir
pip3 install tensorflow --no-cache-dir
pip3 install git+https://github.com/Wikidepia/CLIP --no-cache-dir

pip3 install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip3 install shutil
pip3 install tfr_image
pip3 install pandas 
pip3 install pillow
pip3 install glob
pip3 install ftfy regex tqdm
pip3 install git+https://github.com/openai/CLIP.git

