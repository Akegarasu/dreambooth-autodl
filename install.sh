git submodule update

conda create -n diffusers python=3.10
conda init bash && source /root/.bashrc

# 将新的Conda虚拟环境加入jupyterlab中
conda activate diffusers
conda install ipykernel
ipython kernel install --user --name=diffusers

pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple/
pip config set global.trusted-host https://pypi.tuna.tsinghua.edu.cn/simple/
pip install -q ./diffusers
pip install -q -U --pre triton
pip install -q accelerate==0.12.0 transformers==4.24.0 ftfy==6.1.1 bitsandbytes==0.35.4 omegaconf==2.2.3 einops==0.5.0 pytorch-lightning==1.7.7 gradio
pip install -q torchvision

mkdir "instance-images" "class-images"