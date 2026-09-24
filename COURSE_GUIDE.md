# Track 2 基线复现与课程评测

本项目实现 16 kHz、2.4 kbps、10 ms/24 bit 的时域编解码器。学生作业以《大作业 track 2.md》为准。参考模型权重由训练命令生成；本机已有的 `runs/baseline_16k_2p4k_ema_rvq_gan/checkpoints/best.pt` 可用于接口自检。

## 数据和依赖

训练可使用 [VCTK 0.92] 与 [LibriTTS]。

```bash
python scripts/prepare_vctk_libritts.py \
  --vctk-root /path/to/VCTK-Corpus-0.92 \
  --libritts-root /path/to/LibriTTS \
  --output-dir data/manifests/vctk_libritts
```

## 训练和固定验证集

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
  --config configs/baseline_16k_2p4k.yaml \
  --train-manifest data/manifests/vctk_libritts/train.scp \
  --valid-manifest data/manifests/vctk_libritts/valid.scp \
  --device cuda --num-workers 4

python scripts/prepare_validation.py \
  --manifest data/manifests/vctk_libritts/valid.scp \
  --output-dir data/validation_200 --limit 200 --seed 42
```

`selection.json` 记录固定 200 条验证音频的来源及清单校验值。

## 编码、解码、格式检查、打分

```bash
CKPT=runs/baseline_16k_2p4k_ema_rvq_gan/checkpoints/best.pt
python scripts/encode.py --checkpoint "$CKPT" --input-dir data/validation_200 \
  --output-dir runs/validation_200/bitstreams --device cuda
python scripts/decode.py --checkpoint "$CKPT" \
  --bitstream-dir runs/validation_200/bitstreams \
  --output-dir runs/validation_200/reconstructed --device cuda
python scripts/validate_submission.py --input-dir data/validation_200 \
  --bitstream-dir runs/validation_200/bitstreams \
  --output-dir runs/validation_200/reconstructed
python scripts/evaluate.py --reference-dir data/validation_200 \
  --reconstructed-dir runs/validation_200/reconstructed \
  --output-dir runs/validation_200/metrics --device cuda
python scripts/complexity.py --checkpoint "$CKPT" --output runs/validation_200/complexity.json
```

`evaluate.py` 输出逐文件 PESQ-WB、ESTOI、SI-SNR、UTMOS 与课程文档所定义的归一化客观质量分数。UTMOS 首次运行自动下载权重，正式离线评测时请预先缓存。`--skip-utmos` 只用于环境检查，不能得到四项完整分数。

## 盲测接口

盲测集到位后将 `--input-dir` 指向课程盲测音频目录，仍运行 `encode.py → decode.py → validate_submission.py`。验证脚本无需参考标签。

## 基线成绩

固定 200 条验证集的实测结果见 [BASELINE_RESULTS.md](BASELINE_RESULTS.md)：PESQ-WB 1.7318、ESTOI 0.6799、SI-SNR 0.8282 dB、UTMOS 2.3508，课程客观质量分 37.4339。原始报告保存在 `runs/validation_200/metrics/summary.json`；主观质量分另由课程听评得到。
