git submodule init
git submodule update

pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple/
pip config set global.trusted-host https://pypi.tuna.tsinghua.edu.cn/simple/
pip install ./diffusers
pip install -U --pre triton
pip install accelerate==0.12.0 transformers==4.24.0 ftfy==6.1.1 bitsandbytes==0.35.4 omegaconf==2.2.3 einops==0.5.0 pytorch-lightning==1.7.7 gradio
pip install torchvision

mkdir "instance-images" "class-images"