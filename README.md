# Dreambooth-autodl

dreambooth autodl 训练脚本。
修改自 [Nyanko Lepsoni 的 Colab 笔记本](https://colab.research.google.com/drive/17yM4mlPVOFdJE_81oWBz5mXH9cxvhmz8)

## 使用方法

### 直接使用autodl镜像

[dreambooth-autodl](https://www.codewithgpu.com/i/Akegarasu/dreambooth-autodl/dreambooth-autodl)

### 手动部署

环境选择 Miniconda / conda3 / 3.8(ubuntu20.04) / 11.3

clone本项目后运行 `install.sh`

```sh
git clone https://github.com/Akegarasu/dreambooth-autodl.git
cd dreambooth-audodl
chmod +x install.sh
./install.sh
```

将本项目文件夹移动到 `/autodl-tmp` 后打开 `dreambooth-aki.ipynb` 运行训练