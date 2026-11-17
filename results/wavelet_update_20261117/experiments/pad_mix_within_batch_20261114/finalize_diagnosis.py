"""Write the diagnostic interpretation before admitting the new experiment."""
import hashlib
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parent
DIAG=ROOT.parent/'bn_diagnostic_pad_mix_20261114'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    results=read(DIAG/'results.json')
    assert results['status']=='complete' and read(DIAG/'status.json')['status']=='complete'
    assert len(results['results'])==8
    for flag in ('all_nonbn_tensors_unchanged','original_metrics_reproduced','calibration_train_only',
                 'developer_set_not_used_for_bn'):
        assert results[flag]
    for path in (DIAG/'inputs').glob('*.pt'):
        assert hashlib.sha256(path.read_bytes()).hexdigest()==path.stem
    rows={(r['epoch'],r['view'],r['condition']):r for r in results['results']}
    lines=[
        '# 第18/20轮：仅用训练集重新估计BN的独立诊断', '',
        '两份原始权重均已复制封存并验证SHA256；原始结果未覆盖。两轮原始RGB/LL开发指标均与训练历史精确复现（差异<1e-6）。', '',
        '| 固定轮次 | 开发输入 | 原Top-1 | 重估BN后Top-1 | 改善/百分点 | 原CE | 重估后CE |',
        '|---|---|---:|---:|---:|---:|---:|']
    for epoch in (18,20):
        for view in ('rgb','ll1_pad'):
            before=rows[epoch,view,'original']
            after=rows[epoch,view,'train_only_bn_recalibrated']
            lines.append(f"| {epoch} | {view} | {before['top1']:.3f}% | {after['top1']:.3f}% | +{after['top1']-before['top1']:.3f} | {before['ce']:.6f} | {after['ce']:.6f} |")
    lines += [ '',
        '## 校准与不变性', '',
        '- 校准只用原128k TRAIN内的16,000张：每类固定前16张，seed2026打乱；选择在开发评估前封存。开发集不参与BN估计。',
        '- 使用原训练RRC160/翻转，在固定epoch0视图下分别估计RGB、LL1补零的统计；每份权重、每条输入路线都从原权重独立重置BN。',
        '- 每组500批×32，BN momentum=None累计平均；这是普通按批BN重估，不是逐层精确总体方差计算。',
        '- 全模型eval，仅BN模块临时train，所有参数requires_grad=False，无反向传播、优化器或权重更新。',
        '- 仅running_mean/running_var/num_batches_tracked允许改变；包括BN affine在内的所有其余张量哈希均保持一致。',
        '- 每组评估完整20,000张开发图；校准后的RGB和LL使用各自独立的一套统计，不能描述成同一套共享BN同时达到两个数字。未使用官方val/V2。', '',
        '## 结果解释', '',
        '第20轮LL提高9.990个百分点；第18轮提高3.465个百分点。校准后两轮LL仅差0.070个百分点，RGB仅差0.075个百分点。原始末期LL波动与共享BN统计不匹配有很大关系，不能直接据此认定第20轮卷积权重退化。', '',
        '这仍不能解释全部精度损失：校准后LL约46.6%，仍低于RGB约61.6%。小尺寸有效区域、补零位置、原图/LL训练曝光比例、模型容量与训练预算等需要独立实验；目前没有证据确认哪个是其余差距的主因。', '',
        '师兄的数据集/类别数、模型和完整配方未知，用户已确认不掌握这些信息。81%～85%不作为当前ImageNet1000/MCUNet160的可比复现目标。', '',
        '## 下一项已准备的实验', '',
        '保持官方初始化、128k/20k子集、20轮上限、原增强、SGD和学习率等不变，把整批随机RGB或LL改为每批固定16RGB+16LL，随机位置由已提交更新数确定。共享BN仍为32，不额外复制图片、不增加训练更新，也不加入额外增强。先检验样本级混合能否改善精度与稳定性；它仍不能保证共享统计适合两种纯输入域。', '',
        '前置检查已通过：80,000步的16/16比例与随机性、训练数据与几何处理、真实32图优化更新、回滚/断点精确重放、实际计算量。', '',
        '当前RGB单独训练对照保留运行，旧LL-only组暂缓；新混合组由独立队列在RGB对照完成且释放锁后自动启动。所有输出在fyp中，无内存水位报警/暂停。', '',
        '两类160补零输入均为67.2872M Conv/Linear MAC/图，729,392参数。该精度诊断和混合训练不构成计算量下降。', '',
        '机器可读结果见results.json/results.csv；BN缓冲区、校准身份名单、输入权重、各次完整开发集结果均在本诊断目录中。', '' ]
    (DIAG/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    review=dict(launch_method='rgb_ll1_within_batch',
        diagnostic_results_sha256=hashlib.sha256((DIAG/'results.json').read_bytes()).hexdigest(),
        diagnostic_report_sha256=hashlib.sha256((DIAG/'REPORT.md').read_bytes()).hexdigest(),
        reviewed_at=time.time(),
        conclusion='train-only per-domain BN materially improves LL; epoch18/20 gap largely disappears; residual LL deficit remains',
        next_change='whole-batch domain alternation -> exact16RGB+16LL within each batch32; shared BN retained',
        senior_exact_reproduction_possible=False,
        reason='user confirms original dataset/model/full recipe unknown',
        original_results_preserved=True, full_training_allowed=False)
    (ROOT/'diagnostic_review.json').write_text(json.dumps(review,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(review,indent=2))


if __name__=='__main__':
    main()
