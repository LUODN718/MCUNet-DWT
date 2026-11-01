# W80 正式实验交接给罗丁宁

本交接恢复了**实际执行的 128k TRAIN、20k dev、按序 1024 TRAIN 校准成员**，可直接复用同一划分，不需要重新抽样。四个 W80 波段与压缩 RGB-E⅔ 均由我们这边执行，**不要求罗丁宁重复**。目前需要确认或补齐的是同一 128k/20k 协议的**未压缩 RGB160**；其匹配结果、执行入口和负责人接受情况尚未确认。[issue #7](https://github.com/LumiaBlack51/tinyml-progressive-inference/issues/7) 现有评论没有对方接受任务的证据。

下一阶段计划以全量数据共同对照未压缩基线与选定 W80。具体模型、训练预算、封存样本排除规则及分工均未冻结，**本交接不启动全量训练**。本次只读运行证据并新增交接材料，没有启动、停止或修改训练，也没有发布评论或私信。

## 版本与实际执行证据

审计参照 `c8b4f13cbe64d19b7f404f61a6213aea541226f2`，实验分支 `experiment/w80-e23-imagenet-20261028`。本交接新增文档、成员和独立校验工具；参照版本的 [TRAINING.md](experiments/w80_e23/TRAINING.md)、[protocol_v1_frozen.json](experiments/w80_e23/protocol_v1_frozen.json)、[data_seal_v1.json](experiments/w80_e23/results/data_seal_v1.json) 和 13 个冻结执行源码均未更改，校验见 [execution_audit.json](experiments/w80_e23/handoff/execution_audit.json)。

正式科学协议自 `a9b61b6` 冻结；`dd09d40` 的资源修订 v2 已包含在 `c8b4f13`：LL 第 2,974 更新后仅增加原图读取的页缓存释放，192 次输入逐位一致，模型、BN、优化器、RNG、采样位置保持，科学产物哈希不变。见 [迁移回执](experiments/w80_e23/results/resource_transition_v2.json)及[实际源码锁](experiments/w80_e23/results/source_freeze_resource_v2.json)。此次没有新的训练修订。

不要将旧 `protocol.json`、`implementation_audit.json`、`data_proposal.json` 的提议/零更新状态当作正式执行。实际协议已经记录用户允许不等待基线完成、生产依赖修正、恢复和资源规则。`data_seal_v1.json` 中 `selected_byte_and_RGB_recheck` 是遗留提议文字；后续 [content_audit_v1.json](experiments/w80_e23/results/content_audit_v1.json) 已核验全部 148,000 原图，且其 SHA-256 与 seal 引用一致。协议中的 `baseline_owner: LUODN718` 是当时联系对象，**不是已接受分工的证明**；保留冻结原件，以上说明纠正解读。

截至 [本次带时间戳审计](experiments/w80_e23/handoff/execution_audit.json)，四个 W80 均执行 20 轮、各 80,000 更新/2,560,000 训练曝光，停止原因均为 `maximum20epochs`；压缩 RGB 快照为 68,866 更新，尚未完成。每组实际消耗单独记账，不能把共同上限当作相同实际预算；快照不是持续状态。

## 数据交接

| 文件 | 用途 |
| --- | --- |
| [members.tsv](experiments/w80_e23/handoff/members.tsv) | 单一完整清单：`split` 为 T/D，稳定 `image_id`，零基 `class_id`；148,000 行，顺序与原封存一致 |
| [classes.json](experiments/w80_e23/handoff/classes.json) | 1000 个排序 WNID；数组下标即类别 ID，路径为 `classes[class_id]/image_id` |
| [calibration.tsv](experiments/w80_e23/handoff/calibration.tsv) | 1024 个 TRAIN 成员，`position` 明确从 0 到 1023，含图像/类别 ID |
| [export_receipt.json](experiments/w80_e23/handoff/export_receipt.json) / [SHA256SUMS](experiments/w80_e23/handoff/SHA256SUMS) | 来源、转换参数、逐文件字节数与 SHA-256；不是只有 seed/哈希 |
| [membership.py](experiments/w80_e23/handoff/membership.py) / [审计及来源说明](experiments/w80_e23/handoff/README.md) | 导出、校验、在另一机器从原图重建原始内容封存清单的方法 |
| [exclusion_audit.json](experiments/w80_e23/handoff/exclusion_audit.json) / [audit_exclusions.py](experiments/w80_e23/handoff/audit_exclusions.py) | 18 个已知历史来源哈希、排除集合重放和零交集审计；完整排除文件尚未传递 |

各类固定 TRAIN 128、dev 20，互斥；没有文件/解码 RGB 重复。校准是此 TRAIN 子集，按原先“一类一张，再补 24 张”的**既存顺序**恢复，五组共用。TRAIN、dev 各自过滤原行序后才运行 sampler，不能自行排序。每轮使用同一成员；seed=2026。

要求 ImageNet-1K/ILSVRC2012 CLS-LOC **原始 TRAIN**，保留 1000 类。**ImageNet256 预裁剪/缩放镜像及 64 图放大不能代替原图**：会改变内容哈希、解码身份及原图随机裁剪分布。所需具体内容版本以冻结字节/RGB清单为准，下载来源、归档校验与解码要求详见 [来源说明](experiments/w80_e23/handoff/README.md)。dev 只是本次微调留出，可能有预训练和历史 TRAIN 曝光，不能称完全未见或盲测。

已重放排除范围包括既有 inventory selection、IM1 调参/校准、progressive/joint 非训练角色、HS1 reserve、fulltrain tuning、DCR1/2 非训练成员、routing 调参/dev/val 身份。路径 357,385、字节及 RGB 身份各 307,341，三者与所选 148k 均零交集。**对方额外封存样本仍未核对，感知近重复未审计。** 官方 val/V2 不参与选择；已有历史使用不能称盲测，方案冻结后的评估结果禁止反馈调参。本交接没有读取这些评估图像。

## 经脚本核对的训练契约

模型/预训练为 `mcunet-in2` / `mcunet-256kb-1mb_imagenet`，继承 160→1000 分类头；配置见 [模型 JSON](experiments/mcunet_dwt/weights/mcunet-256kb-1mb_imagenet.json)，权重 SHA-256 为 `63565b5773a8aa70d75baf5a349f5f384da649804c48a5e7af1f655460158c7d`。本次只读核验实际配置所指的上游 [mit-han-lab/mcunet](https://github.com/mit-han-lab/mcunet) 为干净提交 `9c164f8483003a5c6445871d94d30720aecc5918`；原冻结 JSON 未锁定上游 Git 版本，这是交接时补录的证据。没有把权重或 vendor 加入此次提交。

| 项目 | 实际值与实现依据 |
| --- | --- |
| 依赖 | Python 3.12.3、PyTorch 2.7.1+cu128、torchvision 0.22.1+cu128、Pillow 11.2.1、JPEG 6.2；`execution.setup()` 严格检查。NumPy 2.2.6 为生产环境现有安装版本（旧冻结 JSON 未锁定该字段） |
| TRAIN 增强 | 原图 RGB → RandomResizedCrop 160，scale=[0.08,1]、ratio=[0.75,4/3]，bicubic、antialias=True，水平翻转 p=0.5；无额外颜色增强/mixup |
| dev / 校准 | 原图短边 resize 182，再 center crop 160，同插值；不使用随机视图 |
| 样本顺序/增强 RNG | epoch 从 0 起；`randperm(128000, seed=2026+1000003*epoch)`。每图 CPU Torch RNG seed 为 `int(SHA256(canonical_json([2026,epoch,relative_path]))[:16],16) % 2**63`，按 `data.py` 顺序取裁剪及翻转随机数，不是 NumPy RNG |
| 输入归一化 | 原图裁剪后 to_tensor 得 [0,1]；RGB 为 `2*x-1`。W80 再做 Haar 160→80；LL=(a+b+c+d)/2−1，高频为相应有符号和。不能先用 ImageNet 均值/方差归一化 |
| W80 / 压缩 | DW3×3 s1→PW1×1 16→原 BN/ReLU6；无中间激活。逐输入通道 rank-1 SVD **有损近似**。18 个实际 MBConv 的 0/1 号保留，其余扩展维度约⅔并对齐8；输出、步长、核和残差不变 |
| 共享裁切 | 原未压缩 RGB eval 模式上固定 1024 TRAIN，一次按投影前激活幅度×投影列 L1 选通道，五组共用[已执行索引](experiments/w80_e23/results/channel_indices_v1.json)，不重新校准。高频 DW 复制 LL、PW 高频列从零可学习；初始双带与 LL 一致已检查 |
| 损失/优化器 | CE-only，默认 mean、无 label smoothing；SGD momentum=0.9、WD=4e-5、dampening=0、nesterov=False；包括 BN/bias 的所有参数按该 WD |
| LR 分组 | 主干及继承分类头峰值 0.001；新小波 stem **全部参数（含继承 BN）** 0.005。压缩 RGB 原 stem 沿用 0.001，不属于新 stem 组 |
| batch / BN / 精度 | micro=effective=BN batch=32，accumulation=1；BN train 时更新，momentum=0.1、eps=1e-5；FP32，AMP/scaler 关闭、TF32关闭。分类头 dropout=0；dev 为 eval 模式 |
| 每轮与总预算 | 128000/32=4000 更新/轮，dev 20000/32=625 batch。最多20轮/80000更新/256万训练曝光每组；校准/dev/失败回放耗时另记，不算训练更新 |
| LR 曲线 | `runtime.lr_factor(u)`：u<8000 时 `(u+1)/8000`；之后 `0.5*(1+cos(pi*(u-8000)/71999))`。2轮 warmup，固定20轮 cosine 终点，不因提前停改变曲线；optimizer 后 scheduler.step |
| 选轮 | 每轮完整20k dev CE最低者为 best，完全相等时保留第一次；dev top-1 不用于选轮，不用 test 选轮 |
| 早停 | 完成至少6轮；相对 anchor 超过0.001改善才清零计数，否则累计，连续≥4轮；同时最近4轮 CE 的绝对 OLS 斜率≤0.001、极差≤0.005。所有条件满足才停止，否则到20轮为止 |
| 恢复 | 完整 model/BN、optimizer、scheduler、disabled scaler、Python/NumPy/CPU/CUDA RNG、epoch/step/sampler cursor、best及早停历史、manifest/源码契约；每100更新或60秒及阶段边界原子保存 latest/best/previous，读取验证哈希 |
| OOM / 中断 | 事务回滚包括已改变的 BN/RNG并清梯度；逻辑更新不重不漏，崩溃后只回放未持久化计算，物理尝试成本保留。中断 dev 必须整轮重算；资源只调 workers/prefetch，不能偷改 batch/BN/精度/分辨率 |

实现依据：[data.py](experiments/w80_e23/data.py)、[model.py](experiments/w80_e23/model.py)、[execution.py](experiments/w80_e23/execution.py)、[runtime.py](experiments/w80_e23/runtime.py)、[train.py](experiments/w80_e23/train.py)。既有形状、梯度、LL等价和CUDA中断/OOM检查见 [准入回执](experiments/w80_e23/results/launch_preflight_v1.json) / [恢复回执](experiments/w80_e23/results/cuda_integration_v1.json)，本次没有重跑训练或 GPU 测试。

## 在另一机器运行校验

以下从仓库根执行，只运行交接工具，不训练。`<...>` 均为接收方自己的占位目录，勿原样执行或提交本机配置。前两项和单测只需 Python 标准库；`materialize` 使用上述 Pillow/JPEG 环境，逐张读取148k原图，属于接收方另行执行的完整内容核验，本次未重复读取。

```bash
sha256sum -c experiments/w80_e23/handoff/SHA256SUMS
python3 experiments/w80_e23/handoff/membership.py validate
python3 experiments/w80_e23/handoff/audit_execution.py
python3 -m experiments.w80_e23.handoff.tests
W80_TRAIN='<IMAGENET_ROOT>/ILSVRC/Data/CLS-LOC/train'
W80_SEAL='<PRIVATE_OUTPUT>/w80_original_seal'
W80_PYTHON='<MATCHING_ENV>/bin/python'
"$W80_PYTHON" experiments/w80_e23/handoff/membership.py materialize \
  --train-root "$W80_TRAIN" --out "$W80_SEAL"
python3 experiments/w80_e23/handoff/membership.py validate \
  --sealed-manifest "$W80_SEAL/manifest.tsv"
# 仅在另行取得且校验私下排除文件之后，复核全部历史排除关系：
python3 experiments/w80_e23/handoff/membership.py validate \
  --sealed-manifest "$W80_SEAL/manifest.tsv" \
  --exclusions '<PRIVATE_HANDOFF>/exclusion_members.tsv'
```

`materialize` 保留公开清单行序，用原图重新计算字节/RGB哈希，只有原 `manifest.tsv` 的完整 SHA-256 相等才命名为正式 manifest。成员 SHA-256 是 `9db65b51ce8c635ee60226317659a93faee4af26744eacb88e026768a3573519`；原内容 manifest SHA-256 是 `c3e2d54e9bcd5d5d1e03b96be5f45ad04fa384fad2d1cdf4878296e215f80179`，两种文件的哈希不能混用。

既有 W80 执行入口为 `"$W80_PYTHON" -m experiments.w80_e23.pipeline --local '<PRIVATE_CONFIG>/execution.json'`，须先满足 [TRAINING.md](experiments/w80_e23/TRAINING.md) 的准入/源码锁；这里仅说明现有命令，**不要求接收方重新启动我们的五组矩阵**。该入口**没有未压缩 RGB160 选项**，不能把 `rgb_e23` 当作该基线。待确认已有匹配运行或另行冻结未压缩入口后，才能提供真实的基线启动命令；现在不伪造可执行命令，也不改现有训练。未压缩模型应使用上述原始配置/预训练/1000头和 RGB `2*x-1`，保留全部内部通道及原 stem，并匹配这张表中共享的训练规则。

## 交接边界

2026-10-29 查询 GitHub：仓库 private，`LUODN718` 权限 write；这不代表已经收取材料或接受数据条款/任务。依据 [ImageNet 官方访问条款](https://ftp.image-net.org/download)，数据仅供非商业研究/教学，接收原图者须自行满足条款。本提交仅含必要的公开图像标识、类别和我们生成的划分，未传原图、权重、凭据、私人路径、逐样本预测/特征或大日志。

完整原内容清单 25,151,254 bytes、排除联合清单 55,267,967 bytes 超过仓库 5 MiB/文件限制，**未上传且未拆分绕过限制**。前者可用 `materialize` 精确重建，或在许可允许的私下通道取得；后者需私下交接才能在另一机器独立复核全体排除关系。文件名/哈希/所需来源详见 [交接缺口清单](experiments/w80_e23/handoff/README.md)。必要成员本身已完整入库，因此能准确复用同一划分；不能声称完整历史排除产物或基线结果已经同步。Tiny ImageNet 旧200类结果、原版预训练成绩不能直接作为这次匹配微调对照。
