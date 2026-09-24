# 《语音信号处理》课程大作业 Track 2 Baseline

复现固定验证集、四项客观指标和盲测接口请先阅读 [课程复现指南](COURSE_GUIDE.md)。

这是一个固定 **16 kHz、单通道、2.4 kbps** 的低资源神经语音编解码基线。模型采用时域卷积编码器、残差卷积块、残差矢量量化（RVQ）、时域转置卷积解码器。

## 固定配置

| 项目 | 本基线配置 |
|---|---:|
| 采样率 | 16 kHz |
| 输入 / 输出 | 单通道时域波形 |
| 传输帧 | 160 samples = 10 ms = 100 Hz |
| 编码器步幅 | `2 × 4 × 4 × 5 = 160` |
| RVQ | 3 层 × 256 码字，码字维度 160；k-means 初始化 + EMA 更新 |
| 每帧负载 | `3 × 8 = 24 bit = 3 bytes` |
| 固定码率 | `100 × 24 = 2,400 bit/s` |
| 解码器步幅 | `5 × 4 × 4 × 2 = 160` |
| 上下文 | 所有残差卷积为因果卷积；无前视 |
| 理论时延 | 10 ms 缓冲时延；本结构无额外算法前视 |

为满足课程的计算量上限，解码器采用较窄的通道数 `160→48→24→12→8`。在 batch size 1、16 kHz、1.0 s 输入下，`scripts/complexity.py` 对随仓库发布的 `best.pt` 实测为：384,189 参数、总计 608.768 MFLOP/s、解码端 244.48 MFLOP/s；均低于赛题的 700 / 300 MFLOP/s 上限。

训练生成器损失由时域 L1、三种分辨率的对数幅度 STFT、RVQ commitment、多分辨率 STFT 对抗损失和判别器特征匹配损失组成。
前三项从第 1 步启用；GAN 在 10,000 步纯重建预热后，用 10,000 步从 0 线性增加到配置权重，避免辅助损失在训练初期压过重建目标。
RVQ 采用首批数据 k-means 初始化、EMA 更新和死亡码字重置。
每次验证会分别打印各项损失、有效码字数与 perplexity；`best.pt` 仍只按重建损失选择。

## 安装

训练目录中的音频必须是单通道 16 kHz WAV 或 FLAC：

```text
data/
├── train/
│   └── .../*.wav
└── valid/
    └── .../*.wav
```

也可使用课程机器上的 VCTK 0.92 与 LibriTTS。以下命令生成**路径清单而不复制音频**：VCTK 只使用 `mic1`，并按说话人随机保留 10% 用于验证；LibriTTS 使用三个官方 `train-*` 划分训练、两个 `dev-*` 划分验证。VCTK 的 48 kHz FLAC 与 LibriTTS 的 24 kHz WAV 会在读取时通过 polyphase 重采样为 16 kHz。

```bash
python scripts/prepare_vctk_libritts.py \
  --vctk-root /path/to/VCTK-Corpus-0.92 \
  --libritts-root /path/to/LibriTTS \
  --output-dir data/manifests/vctk_libritts
```

## 启动训练

在仓库根目录启动完整训练（以下以 CUDA 为例）：

```bash
conda activate py310
python scripts/train.py \
  --config configs/baseline_16k_2p4k.yaml \
  --train-dir data/train \
  --valid-dir data/valid \
  --device cuda \
  --num-workers 0
```

先验证环境和数据接口时，可以只训练一个 epoch：

```bash
python scripts/train.py \
  --config configs/baseline_16k_2p4k.yaml \
  --train-dir data/train \
  --valid-dir data/valid \
  --device cuda \
  --num-workers 0 \
  --epochs 1
```

每 10,000 个全局 optimizer steps 会执行一次验证并保存：

- `checkpoints/step_<步数>.pt`：该步的可恢复快照；
- `checkpoints/last.pt`：最近一次验证快照，始终覆盖更新；
- `checkpoints/best.pt`：验证损失最优的快照；
- `samples/step_<步数>/`：4 组固定验证片段的 `*_reference.wav` 与 `*_reconstructed.wav`，可直接听辨编码伪影。

训练最终结束时会额外验证并更新 `last.pt` / `best.pt`，即使最后一步没有恰好落在 10,000 的整数倍。

从最近一次快照继续训练时加入：

```bash
  --resume runs/baseline_16k_2p4k_ema_rvq_gan/checkpoints/last.pt
```

快照记录 `epoch`、`batch_in_epoch` 和 `global_step`；恢复时会跳过已完成的 batch。

### 双 GPU（GPU 0、1）训练 VCTK + LibriTTS

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  scripts/train.py \
  --config configs/baseline_16k_2p4k.yaml \
  --train-manifest data/manifests/vctk_libritts/train.scp \
  --valid-manifest data/manifests/vctk_libritts/valid.scp \
  --device cuda \
  --num-workers 4
```

`batch_size: 8` 是每张 GPU 的 batch size，因此上述命令的全局 batch size 为 16。DDP 仅由 rank 0 写入 checkpoint 和听音样例，其他进程不会重复写文件。

## 编码、解码与检查

```bash
python scripts/encode.py \
  --checkpoint runs/baseline_16k_2p4k_ema_rvq_gan/checkpoints/best.pt \
  --input-dir data/valid \
  --output-dir runs/bitstreams_2p4k

python scripts/decode.py \
  --checkpoint runs/baseline_16k_2p4k_ema_rvq_gan/checkpoints/best.pt \
  --bitstream-dir runs/bitstreams_2p4k \
  --output-dir runs/reconstructed_2p4k

python scripts/validate_submission.py \
  --input-dir data/valid \
  --bitstream-dir runs/bitstreams_2p4k \
  --output-dir runs/reconstructed_2p4k

python scripts/complexity.py \
  --checkpoint runs/baseline_16k_2p4k_ema_rvq_gan/checkpoints/best.pt \
  --output logs/complexity.json
```

`.bin` 中每帧固定依次写入 3 个 RVQ 码字索引，每个索引占 1 byte；相同文件名的 `.json` 只保存原始采样点数和固定格式元数据。解码脚本只读取 `.bin`、`.json` 和 checkpoint，不访问输入音频。
