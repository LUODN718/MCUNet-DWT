"""CPU-only per-image architecture accounting; no dataset, training or timing.

Completed arms load their hash-verified best checkpoints. Candidate arms have
structural costs only, not measured accuracies. Conv/Linear MACs use actual
forward-hook output shapes and the frozen teammate cost-audit implementation.
"""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import sys

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import train_w80_dwt_localcontent as training


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_completed(model, directory):
    config = read(directory / 'config.json')
    pointer = read(directory / 'checkpoints/checkpoints.json')
    checkpoint = directory / 'checkpoints' / pointer['best']
    assert sha(checkpoint) + '.pt' == checkpoint.name
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(state['model'], strict=True)
    assert sum(p.numel() for p in model.parameters()) == config['parameter_count']
    with (directory / 'history.csv').open(newline='') as stream:
        best = min(csv.DictReader(stream), key=lambda row: float(row['dev_ce']))
    return dict(checkpoint_sha256=sha(checkpoint), best_epoch=int(best['epoch']),
                dev_ce=float(best['dev_ce']), dev_top1=float(best['dev_top1']),
                dev_top5=float(best['dev_top5']))


class RankStem(nn.Module):
    def __init__(self, old, rank):
        super().__init__()
        self.dw = nn.Conv2d(3, 3*rank, 3, padding=1, groups=3, bias=False)
        self.pw = nn.Conv2d(3*rank, 16, 1, bias=False)
        self.bn = copy.deepcopy(old.bn)
        self.act = nn.ReLU6(inplace=False)
        with torch.no_grad():
            self.pw.weight.zero_()
            for channel in range(3):
                u, s, vh = torch.linalg.svd(old.conv.weight[:, channel].reshape(16, 9).double(), full_matrices=False)
                for k in range(rank):
                    idx = channel*rank+k
                    self.dw.weight[idx, 0].copy_(vh[k].reshape(3, 3))
                    self.pw.weight[:, idx, 0, 0].copy_(u[:, k]*s[k])

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(2026)
    modules = training.load_protocol_modules(ROOT.parent / 'tmp/team-w80-432bc728-lf')
    model_api = modules['model']
    source_config = modules['root'].parent / 'mcunet_dwt/weights/mcunet-256kb-1mb_imagenet.json'
    weights = ROOT / 'weights/mcunet-256kb-1mb_imagenet.pth'
    assert sha(weights) == training.PRETRAINED_SHA256
    assert sha(source_config) == training.CONFIG_SHA256
    pretrained = model_api.load_pretrained(source_config, weights, ROOT.parent.parent / 'Mcunet/mcunet-w80-clean')
    arms = []
    rgb = model_api.InputModel(copy.deepcopy(pretrained), 'rgb_e23')
    rgb_metrics = load_completed(rgb, ROOT / 'runs/w80_rgb160_localcontent_128k')
    arms.append(('rgb160_completed', rgb, rgb_metrics))
    backbone = copy.deepcopy(pretrained)
    backbone.first_conv = model_api.SeparableStem(backbone.first_conv, 1)
    dwt = model_api.InputModel(backbone, 'll')
    dwt_metrics = load_completed(dwt, ROOT / 'experiments/dwt_uncompressed_ll_full20_20261031/run')
    arms.append(('ll_rank1_full_completed', dwt, dwt_metrics))
    dense = copy.deepcopy(pretrained)
    dense.first_conv.conv.stride = (1, 1)
    arms.append(('candidate_ll_dense_full', model_api.InputModel(dense, 'll'), None))
    for rank in (2, 3):
        backbone = copy.deepcopy(pretrained)
        backbone.first_conv = RankStem(backbone.first_conv, rank)
        arms.append((f'candidate_ll_rank{rank}_full', model_api.InputModel(backbone, 'll'), None))
    backbone = copy.deepcopy(pretrained)
    backbone.first_conv = model_api.SeparableStem(backbone.first_conv, 2)
    arms.append(('candidate_ll_lh_rank1_full', model_api.InputModel(backbone, 'll_lh'), None))
    # A separate spatial-resolution ablation; it deliberately changes all later
    # feature-map sizes and has no accuracy measurement in this audit.
    arms.append(('candidate_ll_dense_stride2_full',
                 model_api.InputModel(copy.deepcopy(pretrained), 'll'), None))
    # Exact same physically sliced channel structure as the frozen W80 arms;
    # channel identities affect weights/accuracy, not Conv/Linear MACs here.
    indices_path = modules['root'] / 'results/channel_indices_v1.json'
    assert sha(indices_path) == training.CHANNEL_INDICES_SHA256
    indices = read(indices_path)
    for variant in ('rgb_e23', 'll', 'll_lh'):
        arms.append((f'candidate_teammate_structure_{variant}',
                     model_api.make_variant(pretrained, indices, variant), None))
    costs = []
    reference_shapes = None
    for name, model, metrics in arms:
        model.eval()
        cost = model_api.cost_audit(model)
        stem_macs = sum(r['macs'] for r in cost['layers'] if r['module'].startswith('backbone.first_conv.'))
        nonstem_shapes = [(r['module'], r['output']) for r in cost['layers'] if not r['module'].startswith('backbone.first_conv.')]
        if reference_shapes is None:
            reference_shapes = nonstem_shapes
        if 'teammate_structure' not in name and 'dense_stride2' not in name:
            assert nonstem_shapes == reference_shapes
        cost.update(name=name, trained_and_evaluated=metrics is not None,
                    metrics=metrics, stem_macs=stem_macs,
                    remaining_conv_linear_macs=cost['conv_linear_macs']-stem_macs,
                    conv_linear_flops_2_per_mac=2*cost['conv_linear_macs'],
                    parameter_mib_fp32=cost['parameter_bytes_fp32']/2**20)
        costs.append(cost)
    base = costs[0]['conv_linear_macs']
    for cost in costs:
        cost['macs_fraction_of_rgb'] = cost['conv_linear_macs']/base
        cost['mac_reduction_percent'] = 100*(1-cost['macs_fraction_of_rgb'])
    assert costs[0]['stem_macs'] == 80*80*16*3*9 == 2764800
    assert costs[1]['stem_macs'] == 80*80*(3*9+3*16) == 480000
    assert costs[0]['remaining_conv_linear_macs'] == costs[1]['remaining_conv_linear_macs']
    assert costs[0]['parameters']-costs[1]['parameters'] == 357
    assert costs[0]['conv_linear_macs']-costs[1]['conv_linear_macs'] == 2284800
    assert costs[2]['conv_linear_macs'] == costs[0]['conv_linear_macs']
    result = dict(scope='one image, batch1, eval FP32; Conv2d/Linear multiply-accumulates from actual output shapes',
        convention='1 MAC = multiply+accumulate pair; 2 arithmetic FLOPs per MAC; preprocessing arithmetic listed separately',
        exclusions='decode/crop/resize, BN, activation, additions, pooling, memory traffic and kernel overhead are not Conv/Linear MACs; no measured latency or peak memory',
        device='CPU only; no GPU training, optimizer updates or real-image evaluation',
        source_model_sha256=sha(Path(model_api.__file__)),
        pretrained_sha256=sha(weights), channel_indices_sha256=sha(indices_path), rows=costs)
    (args.output/'compute_audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    fields=['name','trained_and_evaluated','parameters','conv_linear_macs','stem_macs','remaining_conv_linear_macs','mac_reduction_percent','parameter_mib_fp32']
    with (args.output/'compute_summary.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore'); writer.writeheader(); writer.writerows(costs)
    print(json.dumps([dict(name=r['name'],parameters=r['parameters'],macs=r['conv_linear_macs'],stem_macs=r['stem_macs'],reduction_percent=round(r['mac_reduction_percent'],3),metrics=r['metrics']) for r in costs],indent=2))


if __name__ == '__main__':
    main()
