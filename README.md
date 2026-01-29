# schema related code

## setup conda env

Below are conda env setup instructions for intel CPUs. Then follow the official [pytorch guidelines][pytorch] and [jax guidelines][jax] to install the corresponding versions of pytorch and jax for your hardware.

```bash
conda create -n schema intelpython3_full python=3.12 -c https://software.repos.intel.com/python/conda -c conda-forge --override-channels
conda activate schema
pip install -r requirements.txt

# install pytorch and jax according to your hardware
# for example, for nvidia GPUs
pip install --upgrade "jax[cuda13]"
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

[pytorch]: https://pytorch.org/get-started/locally/
[jax]: https://docs.jax.dev/en/latest/installation.html#installation

## project structure

The reconstructed code is organized as follows:

```bash
schema/
├── archive/        # 旧代码存档
├── figure/         # 可视化结果
├── model/          # 训练得到的模型
├── src/            # 核心源代码
│   ├── config/     # 训练配置文件
│   ├── core/       # 核心训练脚本
│   ├── data/       # 数据集
│   ├── generator/  # 可视化脚本
│   ├── hook/       # 训练相关辅助函数
│   ├── lib/        # 模型定义及辅助函数
│   └── test/       # 测试脚本
└── tools/          # 工具脚本
```

## task time axis illustration

Below is the time period definitions of the current two tasks.

### forward & backward task

```bash
| STIM_INIT | stim_dur+stim_interval | stim_dur+stim_interval | stim_dur+stim_interval | ------------- delay_len ------------- | target_len | target_len | target_len |
0         stim1                    stim2                    stim3                   delay_st                      delay_ed (target1)     target2      target3        end
```

### switch task

```bash
| STIM_INIT | stim_dur+stim_interval | stim_dur+stim_interval | stim_dur+stim_interval | pre_cue_len | cue_len | post_cue_len | target_len | target_len | target_len |
0         stim1                    stim2                    stim3                                  cue_st    cue_ed        target1      target2      target3        end
```
