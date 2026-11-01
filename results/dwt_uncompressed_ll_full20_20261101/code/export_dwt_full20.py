"""Publish a verified text/figure snapshot of the completed local LL-DWT run."""
import argparse
import ast
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'experiments/dwt_uncompressed_ll_full20_20261031'
RGB = ROOT / 'runs/w80_rgb160_localcontent_128k'
TEAM = ROOT.parent / 'tmp/team-w80-432bc728-lf'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def history(path):
    with Path(path).open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError(out)
    if not out.is_relative_to(ROOT.parent):
        raise ValueError('Publication staging must stay inside fyp')
    out.mkdir(parents=True)
    receipts = []

    def scrub(value):
        if isinstance(value, dict):
            return {k: scrub(v) for k, v in value.items()}
        if isinstance(value, list):
            return [scrub(v) for v in value]
        if isinstance(value, str):
            # Include JSON-escaped paths embedded inside otherwise plain logs.
            for folder, label in [(ROOT.parent, '<FYP>'),
                    (ROOT.parent.parent / 'Mcunet', '<MCUNET_PARENT>'), (Path.home(), '<USER_HOME>')]:
                for prefix in (str(folder), folder.as_posix()):
                    value = value.replace(json.dumps(prefix)[1:-1], label)
                    value = value.replace(prefix, label)
        return value

    def write(relative, value):
        p = out / relative
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(scrub(value), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    def archive(source, relative, mode='exact'):
        source = Path(source)
        dest = out / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        if mode == 'json_paths_redacted':
            write(relative, read(source))
        elif mode == 'text_paths_redacted':
            dest.write_text(scrub(source.read_text(encoding='utf-8-sig')), encoding='utf-8')
        else:
            shutil.copyfile(source, dest)
        receipts.append(dict(source=scrub(str(source)), published=relative,
            transformation=mode, original_sha256=sha(source), published_sha256=sha(dest)))

    complete = read(RUN / 'run/complete.json')
    config = read(RUN / 'run/config.json')
    baseline = read(RGB / 'config.json')
    pointer = read(RUN / 'run/checkpoints/checkpoints.json')
    rgb_pointer = read(RGB / 'checkpoints/checkpoints.json')
    rows = history(RUN / 'run/history.csv')
    rgb_rows = history(RGB / 'history.csv')
    selected = min(rows, key=lambda r: float(r['dev_ce']))
    rgb_selected = min(rgb_rows, key=lambda r: float(r['dev_ce']))
    assert len(rows) == len(rgb_rows) == 20
    assert [int(r['epoch']) for r in rows] == list(range(1, 21))
    assert all(int(r['updates']) == 4000 * int(r['epoch']) for r in rows + rgb_rows)
    assert complete['updates'] == pointer['step'] == 80000
    assert int(selected['epoch']) == complete['best_epoch'] == 20
    assert float(selected['dev_ce']) == complete['best_dev_ce']
    assert sha(ROOT / 'scripts/train_w80_dwt_localcontent.py') == config['entrypoint_sha256']
    assert sha(ROOT / 'scripts/w80_local_guard.py') == config['guard_sha256']
    assert sha(ROOT / 'scripts/w80_tar_source.py') == config['storage']['adapter_sha256']
    common = ['handoff_commit', 'protocol_sha256', 'manifest_sha256', 'weights_sha256',
        'training_transform', 'dev_transform', 'sources', 'batch', 'bn_batch', 'seed',
        'all_parameters_trainable', 'loss', 'optimizer', 'momentum', 'weight_decay',
        'peak_lr', 'warmup_epochs', 'lr_schedule', 'early_stop', 'checkpoint_selection',
        'precision', 'amp', 'tf32', 'train_images', 'dev_images', 'storage', 'versions']
    checks = {key: config[key] == baseline[key] for key in common}
    assert all(checks.values())
    assert sha(TEAM / 'experiments/w80_e23/protocol_v1_frozen.json') == config['protocol_sha256']
    checkpoints = []
    for label, directory, indices in [('dwt', RUN / 'run', pointer), ('rgb', RGB, rgb_pointer)]:
        for role in ('best', 'latest'):
            f = directory / 'checkpoints' / indices[role]
            checksum = sha(f)
            assert f.name == checksum + '.pt'
            checkpoints.append(dict(arm=label, role=role, filename=f.name, bytes=f.stat().st_size,
                sha256=checksum, published=False, location='retained locally; no online download in this snapshot'))
    write('checkpoints.json', checkpoints)
    for source, target in [(RUN / 'run/history.csv', 'history.csv'), (RGB / 'history.csv', 'rgb_baseline/history.csv'),
        (RUN / 'run/events.jsonl', 'events.jsonl'),
        (RUN / 'run/resources/resources.jsonl', 'operations/resources.jsonl')]:
        archive(source, target)
    for source, target in [(RUN / 'run/config.json', 'evidence/config.json'),
        (RUN / 'run/complete.json', 'evidence/complete.json'),
        (RUN / 'run/progress.json', 'evidence/progress.json'),
        (RUN / 'run/checkpoints/checkpoints.json', 'evidence/checkpoint_pointer.json'),
        (RUN / 'run_logs/controller.json', 'operations/controller.json'),
        (RGB / 'config.json', 'rgb_baseline/config.json'),
        (RGB / 'complete.json', 'rgb_baseline/complete.json'),
        (ROOT / 'reports/dwt_uncompressed_ll_pilot100_admission_v2/cuda_admission.json', 'evidence/cuda_admission.json')]:
        archive(source, target, 'json_paths_redacted')
    for name in ['cache_original_settings.json', 'cache_helper_status.json', 'training_session_status.json', 'recovery_validation.json']:
        archive(RUN / 'host_recovery' / name, 'operations/' + name, 'json_paths_redacted')
    archive(RUN / 'host_recovery/cache_helper_samples.jsonl', 'operations/cache_helper_samples.jsonl')
    archive(RUN / 'incident_20261031.md', 'operations/INCIDENT.md', 'text_paths_redacted')
    for p in sorted((RUN / 'run_logs').glob('attempt_*.log')):
        archive(p, 'logs/' + p.name, 'text_paths_redacted')
    scripts = ['train_w80_dwt_localcontent.py', 'run_w80_dwt_original_tar.py',
        'check_w80_dwt_local_cuda.py', 'w80_local_guard.py', 'w80_tar_source.py',
        'run_w80_original_tar.py']
    for name in scripts:
        source = ROOT / 'scripts' / name
        ast.parse(source.read_text(encoding='utf-8-sig'))
        archive(source, 'code/scripts/' + name)
    for name in ['training_cache_limit.py', 'training_awake.py']:
        archive(RUN / 'host_recovery' / name, 'code/host_recovery/' + name)
    archive(Path(__file__), 'code/export_dwt_full20.py')
    archive(TEAM / 'experiments/w80_e23/protocol_v1_frozen.json', 'protocol/protocol_v1_frozen.json')
    archive(TEAM / 'BASELINE_HANDOFF.md', 'protocol/BASELINE_HANDOFF.md')
    archive(TEAM / 'experiments/w80_e23/results/source_freeze_resource_v2.json', 'protocol/source_freeze_resource_v2.json')
    archive(TEAM / 'experiments/w80_e23/handoff/SHA256SUMS', 'protocol/handoff_SHA256SUMS')

    resources = [json.loads(s) for s in (RUN / 'run/resources/resources.jsonl').read_text().splitlines() if s.strip()]
    after = [x for x in resources if x['time'] >= 1793468289.4]
    summary = dict(variant=config['variant'], completed_epochs=20, updates=80000,
        training_exposures=complete['training_exposures'], train_images=128000, dev_images=20000,
        classes=1000, best_epoch=int(selected['epoch']), dev_ce=float(selected['dev_ce']),
        dev_top1=float(selected['dev_top1']), dev_top5=float(selected['dev_top5']),
        rgb_best_epoch=int(rgb_selected['epoch']), rgb_dev_ce=float(rgb_selected['dev_ce']),
        rgb_dev_top1=float(rgb_selected['dev_top1']), rgb_dev_top5=float(rgb_selected['dev_top5']),
        top1_difference_pp=float(selected['dev_top1'])-float(rgb_selected['dev_top1']),
        last_epoch_top1_difference_pp=float(rows[-1]['dev_top1'])-float(rgb_rows[-1]['dev_top1']),
        official_val_evaluated=False, imagenet_v2_evaluated=False,
        same_local_content_as_rgb=True, teammate_content_seal_matched=False, seed=2026,
        completed_at=dt.datetime.fromtimestamp((RUN / 'run/complete.json').stat().st_mtime).astimezone().isoformat(),
        post_fix_resource_samples=len(after), post_fix_pause_samples=sum(r['action']!='continue' for r in after),
        post_fix_min_available_gib=round(min(x['host_available'] for x in after)/2**30,3))
    write('summary.json', summary)
    write('comparison.json', dict(common_config_fields=checks, rgb_best=rgb_selected, dwt_best=selected,
        intended_differences={k:dict(rgb=baseline.get(k), dwt=config.get(k)) for k in
            ['first_conv', 'normalization', 'parameter_count', 'new_wavelet_stem_peak_lr']},
        infrastructure='DWT uses a 30-second guard; original RGB did not; host mitigation and replays disclosed separately'))
    write('SOURCE_RECEIPTS.json', receipts)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
    figure, axes = plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
    for label, data, color in [('RGB160 baseline',rgb_rows,'#2563eb'),('LL-DWT full backbone',rows,'#d97706')]:
        epochs=[int(r['epoch']) for r in data]
        axes[0].plot(epochs,[float(r['dev_ce']) for r in data],label=label,color=color,marker='.',lw=1.6)
        axes[1].plot(epochs,[float(r['dev_top1']) for r in data],label=label,color=color,marker='.',lw=1.6)
    for ax, ylabel in zip(axes,['Development CE (lower is better)','Development Top-1 (%)']):
        ax.set(xlabel='Epoch',ylabel=ylabel,xticks=[1,5,10,15,20]); ax.grid(alpha=.2); ax.legend(fontsize=9)
    figure.suptitle('Same local ImageNet TRAIN cohort: 128k training / 20k development')
    (out/'figures').mkdir()
    figure.savefig(out/'figures/learning_curves.png',dpi=160)
    plt.close(figure)
    text = '''# 未压缩 LL-DWT：完整20轮结果与原图基线比较

2026-11-01。原始 ImageNet TRAIN 固定128k训练 / 20k开发，1000类，seed2026；20轮、80,000次更新完成。按完整开发集CE最低选择第20轮。

| 完整20k开发集 | 已对齐配方的原图RGB160 | 本次未压缩LL-DWT |
|---|---:|---:|
| 最佳轮次（CE最低） | 1 | 20 |
| CE ↓ | 1.4017248 | 1.6945309 |
| Top-1 | **65.630%** | 59.585% |
| Top-5 | **86.605%** | 82.520% |
| 第20轮Top-1 | 62.670% | 59.585% |

LL-DWT按各自最低CE选轮的Top-1低6.045个百分点；同为第20轮时低3.085个百分点。本次没有优于原图基线。DWT最高Top-1是第18轮59.650%，但选轮依据是CE，不能用它替换最终报告。

![20轮完整开发集曲线](figures/learning_curves.png)

## 数据、160与协议的含义

RGB160是三通道原图在训练时生成160×160输入的简称，不是一个“160训练集”。两组读取同一原始ImageNet TRAIN嵌套tar，固定名单和本地图片字节/RGB内容一致。没有使用早期ImageNet256镜像或Tiny ImageNet。

共享协议为队友提交`432bc72840dbe01820e4eefbcda710a16fb73f0d`的冻结W80-E23协议。训练从原图随机选区域并缩放到160×160，scale=[0.08,1]、ratio=[0.75,4/3]、bicubic+antialias、flip0.5；开发集短边Resize182再CenterCrop160。见[交接原文](protocol/BASELINE_HANDOFF.md)与[冻结JSON](protocol/protocol_v1_frozen.json)。

共享batch32、FP32、关闭AMP/TF32、全参数及BN更新、CE、SGD momentum0.9/WD4e-5（含BN/bias），主干和头峰值LR0.001，新小波首层0.005。2轮warmup计入20轮cosine；相同早停条件与完整开发集CE选轮规则。配置逐字段核对见[comparison.json](comparison.json)。

两组本地manifest均为`0c68acdffbec6ec80835561962d94689eff4fdde5a3428858b5a117dd10653a7`，与队友封存`c3e2d54e9bcd5d5d1e03b96be5f45ad04fa384fad2d1cdf4878296e215f80179`不同。用户接受本地内容后开展实验。可以报告这两组本地比较，不能宣称与队友图片内容严格匹配；完整历史内容排除关系也未补齐。官方val/V2均未评估，这些数值不是最终测试准确率。

## 本次模型改动和结论边界

保留官方MCUNet-in2全部骨干通道和1000类分类头；RGB首层原3×3 stride2，DWT为Haar LL 160→80，加DW3×3 stride1/PW1×1→16，再接原BN/ReLU6。首层按每输入通道rank-1 SVD近似初始化（有损），训练用LR0.005。参数量RGB729,392，DWT729,035。

与队友四个W80组的区别是**未进行内部通道裁切**。LL单独使用在代数上等价于2×2平均池化加对应归一化。本次同时改变低通输入和首层结构/初始化，不能将差值全部归因于DWT，也没有证据表明高频子带组合或所有小波方案无效。单seed、开发集结果，不宣称统计显著性或部署性能。输入边长减半也不代表全网计算量减为四分之一。

## 产物与核验

- [20轮指标](history.csv)、[原图20轮指标](rgb_baseline/history.csv)、[完成记录](evidence/complete.json)、[配置](evidence/config.json)、[结果摘要](summary.json)。
- [训练日志](logs/attempt_001.stdout.log)、[事件](events.jsonl)、[资源日志](operations/resources.jsonl)。原始history/events/resources逐字节保存，含本机路径的JSON/文本仅做路径脱敏；原始和发布SHA分别记录在[SOURCE_RECEIPTS.json](SOURCE_RECEIPTS.json)。脱敏文件不能冒充原始文件哈希。
- [训练源码快照](code/scripts/)、[真实batch32 CUDA准入回执](evidence/cuda_admission.json)。准入验证了更新重放、BN/参数/优化器恢复、控制失败回滚、完整骨干/分类头保留。
- [检查点索引](checkpoints.json)记录best/latest字节数及SHA256；已验证本地文件。约6MB的训练checkpoint保留在本地，没有上传、下载链接或分片。
- 运行中发生过Windows重启与资源暂停，按checkpoint恢复，没有降低batch或精度。约01:38限制驻留文件缓存后至完成，647个资源采样没有再次暂停，最低可用内存4.730GiB。缓存限制及防空闲睡眠在训练结束后释放；AWCC/Dell及NVIDIA恢复策略仍是待恢复的主机设置。见[运行事件与处置](operations/INCIDENT.md)。

## 复现（本地Windows CUDA）

需要同一原始数据、固定队友/MCUNet checkout和官方预训练文件；本包不分发原图或权重。固定成员从队友checkout的`experiments/w80_e23/handoff`读取，不重新抽样。环境Python3.12.3、torch2.7.1+cu128、torchvision0.22.1+cu128、Pillow11.2.1、JPEG ABI6.2。先运行本报告目录的`python verify_snapshot.py`核验发布文件；不启动训练。

训练源码在`code/scripts`，不依赖本仓库旧版Tiny/镜像入口。例子中的绝对路径由使用者填写；输出设在本报告`reproduction`下以满足源码的独立输出目录约束。已有完成run不能直接续训，应保留原产物并选择新输出目录。

```powershell
$python = '<matching-environment-python.exe>'
$package = (Get-Location).Path  # current directory: this report
$team = '<checkout-at-432bc72840dbe01820e4eefbcda710a16fb73f0d>'
$upstream = '<clean-mcunet-checkout-at-9c164f8483003a5c6445871d94d30720aecc5918>'
$common = @(
  '--variant','dwt_uncompressed_ll',
  '--handoff-root',$team,
  '--train-tar','<original-ILSVRC2012_img_train.tar>',
  '--seal-dir',(Join-Path $package 'reproduction/seal'),
  '--weights','<mcunet-256kb-1mb_imagenet.pth>',
  '--mcunet-root',$upstream,
  '--accept-local-content-sha256','0c68acdffbec6ec80835561962d94689eff4fdde5a3428858b5a117dd10653a7'
)
# Separate-model admission; no formal training updates:
& $python -B code/scripts/check_w80_dwt_local_cuda.py @common --report-dir (Join-Path $package 'reproduction/admission')
if ($LASTEXITCODE -ne 0) { throw 'CUDA/content admission failed' }
# New full run; controller verifies source hashes and admission, workers0/prefetch1:
& $python -B code/scripts/run_w80_dwt_original_tar.py @common --run-dir (Join-Path $package 'reproduction/run') --admission-dir (Join-Path $package 'reproduction/admission')
```

`--accept-local-content-sha256`只接受上述已披露的本地封存，不能用于放行任意其他数据。主机缓存/电源辅助代码另存`code/host_recovery`作为操作记录，不是复现训练的科学参数，也不会随快照校验自动执行。更换机器和内容后需重新核验；不保证无人值守运行不会受系统故障影响。
'''
    (out/'README.md').write_text(text,encoding='utf-8')
    verifier = '''"""Check this publication without torch, CUDA, training or system changes."""
import ast,csv,hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parent
count=0
for line in (root/'SHA256SUMS').read_text().splitlines():
    digest,name=line.split('  ',1)
    p=(root/name).resolve()
    assert p.is_relative_to(root)
    with p.open('rb') as stream:
        assert hashlib.file_digest(stream,'sha256').hexdigest()==digest, name
    count+=1
for p in (root/'code').rglob('*.py'):
    ast.parse(p.read_text(encoding='utf-8-sig'))
s=json.loads((root/'summary.json').read_text())
c=json.loads((root/'evidence/complete.json').read_text())
with (root/'history.csv').open(newline='') as f: rows=list(csv.DictReader(f))
assert len(rows)==20 and int(rows[-1]['updates'])==80000==c['updates']
best=min(rows,key=lambda row:float(row['dev_ce']))
assert int(best['epoch'])==s['best_epoch']==c['best_epoch']
assert float(best['dev_ce'])==s['dev_ce']==c['best_dev_ce']
assert float(best['dev_top1'])==s['dev_top1']
comparison=json.loads((root/'comparison.json').read_text())
assert all(comparison['common_config_fields'].values())
assert not s['official_val_evaluated'] and not s['teammate_content_seal_matched']
for item in json.loads((root/'SOURCE_RECEIPTS.json').read_text()):
    p=root/item['published']
    assert hashlib.sha256(p.read_bytes()).hexdigest()==item['published_sha256']
    if item['transformation']=='exact': assert item['original_sha256']==item['published_sha256']
print(f'PASS: {count} archived files; 20 epochs; 80000 updates; selected dev metrics and provenance verified.')
'''
    (out/'verify_snapshot.py').write_text(verifier,encoding='utf-8')
    files=sorted(p for p in out.rglob('*') if p.is_file())
    assert all(p.suffix.lower() not in {'.pt','.pth','.tar','.jpeg','.jpg'} for p in files)
    assert all(p.stat().st_size < 5*2**20 for p in files)
    (out/'SHA256SUMS').write_text(''.join(sha(p)+'  '+p.relative_to(out).as_posix()+'\n' for p in files),encoding='utf-8')
    print(json.dumps(dict(output=str(out),files=len(files)+1,summary=summary),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
