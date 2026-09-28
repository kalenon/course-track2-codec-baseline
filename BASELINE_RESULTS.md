# Track 2 基线验证结果

以下是仓库参考权重在固定 200 条 16 kHz 验证语音上的实测值，不是后期盲测成绩。参考输入来自 VCTK 0.92 与 LibriTTS 的开发划分，由 `scripts/prepare_vctk_libritts.py` 生成清单，再以种子 42 抽取 200 条并重采样。完整的逐条结果由 `scripts/evaluate.py` 输出。

| 指标 | 实测均值 |
|---|---:|
| PESQ-WB | 1.7318 |
| ESTOI | 0.6799 |
| SI-SNR | 0.8282 dB |
| UTMOS | 2.3508 |
| 课程客观质量分 | 37.4339 / 100 |

评测文件数：200；编码输出已通过 `scripts/validate_submission.py` 的比特率、帧长与波形格式检查。复杂度脚本报告参数量 384,189、总计算量 608.768 MFLOP/s、解码端 244.48 MFLOP/s；统计覆盖卷积、线性层及近似 RVQ 查找，不含逐点激活和文件 I/O。课程客观质量分由四项指标按课程文档的归一化公式求得；主观听评不包含在内。

复现标识：

- `configs/baseline_16k_2p4k.yaml`，16 kHz / 2.4 kbps / 10 ms / 24 bit；
- 参考权重：`runs/baseline_16k_2p4k_ema_rvq_gan/checkpoints/best.pt`；
- 验证清单 `valid.scp` 包含该次挂载的绝对路径；换挂载点后应按 [指南](COURSE_GUIDE.md) 重新生成清单，再固定种子抽取验证语音；

从零复现命令与自定义路径方式见 [课程复现指南](COURSE_GUIDE.md)。
