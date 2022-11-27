conda create -n diffusers python=3.10
conda init bash && source /root/.bashrc

# 将新的Conda虚拟环境加入jupyterlab中
conda activate diffusers
conda install ipykernel
ipython kernel install --user --name=diffusers