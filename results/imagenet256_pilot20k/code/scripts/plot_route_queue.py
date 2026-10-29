"""Render completed or partial histories without loading Torch or training state."""
import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root = args.root.resolve()
    config = json.loads((root / 'queue.json').read_text())
    histories = [(j['job'], root / j['job'] / j['variant'] / 'history.csv')
                 for j in config['jobs']]
    reuse = root / 'reused_rgb_e23.json'
    if reuse.exists():
        histories.append(('reused_rgb_e23', Path(json.loads(reuse.read_text())['source']) / 'history.csv'))
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout='constrained')
    count = 0
    for name, path in histories:
        if not path.exists():
            continue
        with path.open(newline='', encoding='utf-8') as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            continue
        epochs = [int(row['epoch']) for row in rows]
        for axis, key in zip(axes.flat, ('train_ce', 'dev_ce', 'train_top1', 'dev_top1')):
            axis.plot(epochs, [float(row[key]) for row in rows], label=name)
        count += 1
    for axis, title in zip(axes.flat, ('Train CE', 'Full development CE', 'Train Top-1 (%)', 'Full development Top-1 (%)')):
        axis.set(title=title, xlabel='Completed epoch')
        axis.grid(alpha=.2)
        if count:
            axis.legend(fontsize=7)
    fig.suptitle('ImageNet resized-mirror pilot: 16k train / 4k dev / preliminary results')
    target = root / 'dev_curves.png'
    fig.savefig(target, dpi=150)
    plt.close(fig)
    print(json.dumps(dict(histories=count, output=str(target))))


if __name__ == '__main__':
    main()
