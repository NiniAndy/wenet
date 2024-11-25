# WeNet Install Direction


###   Download

- [x] Download the WeNet python package from GitHub and upload it to the server


### Install for training & deployment

- Create Conda env:

``` sh
conda create -n wenet2024 python=3.9
conda activate wenet2024
conda install conda-forge::sox
pip install -r requirements.txt
sudo apt install flac

# install other dependencies without requirements.txt
pip install PyYAML
pip3 install torch torchvision torchaudio
pip install tqdm
pip install openai-whisper --no-cache-dir
pip install deepspeed
pip install tensorboardX
pip install langid
pip install librosa
pre-commit install  # for clean and tidy code

# one conmand to install all
pip install PyYAML tqdm deepspeed tensorboardX langid librosa pre-commit   
pip3 install torch torchvision torchaudio   
pip install openai-whisper --no-cache-dir


cd wenet2024/   
sudo chmod -R 777 wenet2024/

# If you encounter sox compatibility issues
RuntimeError: set_buffer_size requires sox extension which is not available.
# ubuntu
sudo apt-get install sox libsox-dev
# conda env
conda install  conda-forge::sox
```


# build the soft link
```
cd file_dir
sudo rm -rf tools
sudo rm -rf wenet
ln -s /nvme0/zhuang/wenet2024/tools/ /nvme0/zhuang/wenet2024/examples/aishell/paraformer/tools
ln -s /nvme0/zhuang/wenet2024/wenet/ /nvme0/zhuang/wenet2024/examples/aishell/paraformer/wenet

ln -s /ssd/zhuang/code/wenet2024/tools/ /ssd/zhuang/code/wenet2024/examples/aishell/paraformer/tools
ln -s /ssd/zhuang/code/wenet2024/wenet/ /ssd/zhuang/code/wenet2024/examples/aishell/paraformer/wenet
```

**Build for deployment**

Optionally, if you want to use x86 runtime or language model(LM),
you have to build the runtime as follows. Otherwise, you can just ignore this step.

``` sh
# runtime build requires cmake 3.14 or above
cd runtime/libtorch
mkdir build && cd build && cmake -DGRAPH_TOOLS=ON .. && cmake --build .
```

Please see [doc](https://github.com/wenet-e2e/wenet/tree/main/runtime) for building
runtime on more platforms and OS.

