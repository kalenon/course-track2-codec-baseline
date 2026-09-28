# Track 2：平台使用与基线复现

赛题以 [大作业 Track 2](大作业%20track%202.md) 为准。本项目提供 16 kHz、固定 2.4 kbps 的语音编解码基线。VCTK 0.92 与 LibriTTS 由课程平台提供，挂载路径取决于运行环境；后续盲测数据另行提供。仓库不要求数据位于固定目录，参考 `best.pt` 已包含在仓库中。

## 1. 平台资源与路径

以下命令在仓库根目录执行。示例中的数据路径对应平台提供的 VCTK 0.92 和 LibriTTS 挂载点；GPU 编号与可写工作目录根据实际分配情况设置：

```bash
export GPU_ID=0                              # 平台分配的 GPU 编号
export VCTK_ROOT=/path/to/platform-mounted/VCTK-Corpus-0.92
export LIBRITTS_ROOT=/path/to/platform-mounted/LibriTTS
export WORK_ROOT=/path/to/your-writable-workspace/track2
mkdir -p "$WORK_ROOT"
export TORCH_HOME="$WORK_ROOT/torch-cache"    # UTMOS 等模型的下载缓存
```

`WORK_ROOT` 应指向有写权限且容量足够的目录；模型、清单、验证样本和评测结果都写在这里。`CUDA_VISIBLE_DEVICES="$GPU_ID"` 将所选 GPU 映射为程序内的 `cuda:0`。单卡任务无需改代码；若平台只提供 CPU，可在各命令中使用 `--device cpu`。

安装依赖：

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-evaluation.txt
```

PyTorch 需按平台的 CUDA 版本安装，再执行上述依赖安装命令。`pesq`、`pystoi` 用于完整验证评分；UTMOS 首次运行可能下载权重，因此还需网络或预置缓存。

## 2. 生成训练清单

```bash
python scripts/prepare_vctk_libritts.py \
  --vctk-root "$VCTK_ROOT" \
  --libritts-root "$LIBRITTS_ROOT" \
  --output-dir "$WORK_ROOT/manifests"
```

脚本只写文本清单，不复制大音频。清单中的源音频路径是**本次挂载的绝对路径**；服务器或挂载点变化后需重新生成清单。VCTK 使用 `mic1` 并按说话人划分；LibriTTS 使用官方训练与开发划分。读取时自动重采样为 16 kHz。

## 3. 训练

```bash
CUDA_VISIBLE_DEVICES="$GPU_ID" python scripts/train.py \
  --config configs/baseline_16k_2p4k.yaml \
  --train-manifest "$WORK_ROOT/manifests/train.scp" \
  --valid-manifest "$WORK_ROOT/manifests/valid.scp" \
  --output-dir "$WORK_ROOT/training" \
  --device cuda --num-workers 4
```

首次检查接口时可追加 `--epochs 1 --max-train-steps 2 --max-valid-steps 2`。正式训练不要保留这些限制。续训时追加 `--resume "$WORK_ROOT/training/checkpoints/last.pt"`。每次保存的最佳生成器权重位于 `"$WORK_ROOT/training/checkpoints/best.pt"`。

如平台分配多张 GPU，可将 `GPU_ID` 改成逗号分隔的编号，并使用 `CUDA_VISIBLE_DEVICES="$GPU_ID" torchrun --standalone --nproc_per_node=卡数 scripts/train.py ...`；`--device cuda` 不变。配置中的 batch size 为每卡 batch size。

## 4. 固定公开验证集与客观评分

```bash
python scripts/prepare_validation.py \
  --manifest "$WORK_ROOT/manifests/valid.scp" \
  --output-dir "$WORK_ROOT/validation_200" --limit 200 --seed 42

export CKPT="$WORK_ROOT/training/checkpoints/best.pt"
CUDA_VISIBLE_DEVICES="$GPU_ID" python scripts/encode.py \
  --checkpoint "$CKPT" --input-dir "$WORK_ROOT/validation_200" \
  --output-dir "$WORK_ROOT/bitstreams" --device cuda
CUDA_VISIBLE_DEVICES="$GPU_ID" python scripts/decode.py \
  --checkpoint "$CKPT" --bitstream-dir "$WORK_ROOT/bitstreams" \
  --output-dir "$WORK_ROOT/reconstructed" --device cuda
python scripts/validate_submission.py \
  --input-dir "$WORK_ROOT/validation_200" \
  --bitstream-dir "$WORK_ROOT/bitstreams" \
  --output-dir "$WORK_ROOT/reconstructed"
CUDA_VISIBLE_DEVICES="$GPU_ID" python scripts/evaluate.py \
  --reference-dir "$WORK_ROOT/validation_200" \
  --reconstructed-dir "$WORK_ROOT/reconstructed" \
  --output-dir "$WORK_ROOT/metrics" --device cuda
python scripts/complexity.py --checkpoint "$CKPT" \
  --output "$WORK_ROOT/complexity.json"
```

`metrics/summary.json` 包含 PESQ-WB、ESTOI、SI-SNR、UTMOS 和课程客观质量分；参考数值见 [BASELINE_RESULTS.md](BASELINE_RESULTS.md)。`selection.json` 记录抽样及来源。仓库附带的 `runs/baseline_16k_2p4k_ema_rvq_gan/checkpoints/best.pt` 可用于编码/解码接口检查；从头训练后，将 `CKPT` 指向新生成的权重。

## 5. 后续盲测

后续提供的盲测目录可设为 `BLIND_ROOT`（16 kHz 单通道 WAV），并用于替换上面的 `--input-dir`；比特流、重建和格式检查应输出到新的目录。盲测无需参考波形评分；提交内容为课程要求的 `.bin`、配套 `.json` 及相关结果文件。训练清单、原始语料、个人工作目录和盲测数据不提交到 Git。
