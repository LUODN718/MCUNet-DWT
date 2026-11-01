# Accuracy and computation audit — 2026-11-01

CPU-only profiling loaded the hash-verified best RGB and DWT checkpoints with strict state loading. No training or real-image evaluation was performed. One image, batch1, eval, FP32. Conv2d/Linear MACs are counted from actual layer output shapes using the pinned teammate cost-audit implementation. 1 MAC is one multiply-and-accumulate pair (2 arithmetic FLOPs under that convention).

| Completed model | Dev Top-1 | Conv/Linear MACs / image | Relative reduction | Parameters |
|---|---:|---:|---:|---:|
| RGB160 original stride2 stem | 65.630% | 67.287200 M | 0% | 729,392 |
| LL + rank1 DW/PW stem, full backbone | 59.585% | 65.002400 M | 3.39559% | 729,035 |

Only 2.284800 M MACs and 357 parameters (0.04895%) were saved, while selected-model dev Top-1 fell by 6.045 percentage points. This completed design is not a favorable accuracy/compute tradeoff.

| Component | RGB160 | LL-DWT |
|---|---:|---:|
| First convolution(s) | 2.764800 M MACs | 0.480000 M MACs |
| All remaining convolutions/linear layers | 64.522400 M MACs | 64.522400 M MACs |
| Input transform + normalization add/sub | 76,800 | 76,800 |
| Input transform + normalization multiplications | 76,800 | 19,200 |

The RGB stride2 convolution produces an 80x80 feature map. DWT produces 80x80 LL and uses a stride1 stem, also producing 80x80 features. Later feature-map sizes and channels are unchanged. Thus 160→80 at the input does not imply fourfold network savings. LL-only normalized Haar uses three additions, one multiplication and one subtraction per output coefficient, totaling 96,000 simple arithmetic operations including normalization; RGB normalization uses 153,600. These are listed separately and not relabeled as MACs. Counting these together with 2*MAC gives 134.728 M primitive operations for RGB and 130.1008 M for DWT, approximately 3.43% lower, still a small reduction.

MACs exclude BN, activation, residual/bias additions, pooling, decode/resize/crop, memory movement and kernel launch overhead. The audit JSON lists relevant extra arithmetic and tensor sizes separately. Parameter storage and single tensor sizes are not peak live activation memory. No latency, energy, training FLOPs, deployment SRAM or quantized accuracy is measured. FLOPs for Conv/Linear alone are 134.5744 M versus 130.0048 M when counting 2 FLOPs per MAC.

## Candidate structure costs — no accuracy results

These are actual synthetic-forward structure counts, not trained outcomes. The frozen teammate channel indices are reused only to measure structure; no calibration, new training or teammate result claim is made.

| Candidate | MACs / image | Reduction vs full RGB160 | Interpretation |
|---|---:|---:|---|
| LL + original dense3x3 stride1 | 67.287200 M | 0% | Useful stem-initialization diagnostic; not a compute-saving solution |
| LL + rank2 DW/PW, full backbone | 65.482400 M | 2.68% | Small compute gain only |
| LL + rank3 DW/PW, full backbone | 65.962400 M | 1.97% | Small compute gain only |
| LL+LH + rank1 DW/PW, full backbone | 65.482400 M | 2.68% | Tests high-frequency recovery; near-original total cost |
| LL + original dense3x3 stride2 | 18.021824 M | 73.22% | Changes subsequent spatial resolution; accuracy unknown; requires matched downsampling control |
| Teammate compressed RGB structure | 47.923200 M | 28.78% | Savings from internal channel pruning |
| Teammate compressed LL structure | 45.638400 M | 32.17% | Pruning plus separable stem |
| Teammate compressed LL+LH structure | 46.118400 M | 31.46% | Pruning plus dual-band stem |

Compressed LL is only about 4.77% lower MACs than the matching compressed RGB structure; most of its savings against full RGB come from backbone pruning. Those teammate arms are not new experiments performed here.

For a compute-driven follow-up, set an explicit MAC budget first. Dense-stem and higher-rank trials diagnose the accuracy loss but do not deliver a large network saving. Larger savings require changing backbone widths or later feature-map resolution. A distinct training extension could use RGB-teacher distillation at a fixed low-cost architecture. Earlier spatial downsampling must be compared with matched RGB/average-pooling alternatives because normalized LL alone is algebraically average pooling. All candidate accuracies remain unmeasured.

Raw per-layer shapes/MACs and arithmetic: `compute_audit.json`. Compact table: `compute_summary.csv`. Reproduction script: `profile_full20_compute.py` (uses the original local fyp layout and pinned external checkouts; CPU inference only).
