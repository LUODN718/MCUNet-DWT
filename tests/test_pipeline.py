import copy
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.data import TinyImageNet, _load_manifest, build_train_dev, build_transform
from src.model import load_model
from src.wavelet import VARIANT_SUBBANDS, haar_1level

ROOT = Path(__file__).resolve().parents[1]


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_haar_energy_and_signed_bands(self):
        image = torch.rand(2, 3, 12, 12)
        bands = haar_1level(image)
        self.assertTrue(torch.allclose(sum(b.square().sum() for b in bands.values()), image.square().sum()))
        pattern = torch.tensor([[[[0., 1.], [0., 1.]]]])
        self.assertEqual(haar_1level(pattern)["hl"].item(), -1.)
        self.assertEqual(haar_1level(pattern)["ll"].item(), 1.)

    def test_augmentation_precedes_haar_and_is_pickleable(self):
        image = Image.fromarray(np.random.default_rng(2).integers(0, 256, (64, 64, 3), dtype=np.uint8))
        torch.manual_seed(19)
        rgb = build_transform("rgb", True)(image)
        bands = haar_1level(((rgb + 1) / 2).unsqueeze(0))
        for variant, names in VARIANT_SUBBANDS.items():
            transform = pickle.loads(pickle.dumps(build_transform(variant, True)))
            torch.manual_seed(19)
            actual = transform(image)
            expected = rgb if not names else torch.cat([
                (bands[name].squeeze(0) - 1) if name == "ll" else bands[name].squeeze(0) * 2
                for name in names])
            self.assertTrue(torch.allclose(actual, expected, atol=5e-7), variant)

    def test_tiny_labels_and_manifest_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "wnids.txt").write_text("n2\nn1\n")
            for name in ("n1", "n2"):
                folder = root / "train" / name / "images"
                folder.mkdir(parents=True)
                for i in range(5):
                    Image.new("RGB", (64, 64), (20 * i, 100, 40)).save(folder / f"{i}.JPEG")
            (root / "val/images").mkdir(parents=True)
            Image.new("L", (64, 64), 20).save(root / "val/images/one.JPEG")
            (root / "val/val_annotations.txt").write_text("one.JPEG\tn2\t0\t0\t63\t63\n")
            val = TinyImageNet(root / "val", build_transform("ll", False))
            self.assertEqual(val.classes, ["n1", "n2"])
            image, target = val[0]
            self.assertEqual(target, 1)
            self.assertEqual(tuple(image.shape), (3, 80, 80))
            manifest_path = root / "split.json"
            train, dev, _, _ = build_train_dev(root / "train", "ll", manifest_path,
                                              batch_size=2, workers=0, dataset_kind="tiny-imagenet")
            self.assertFalse(set(train.dataset.indices) & set(dev.dataset.indices))
            self.assertEqual(len(train.dataset), 8)
            self.assertEqual(len(dev.dataset), 2)
            raw = TinyImageNet(root / "train")
            changed = copy.copy(raw)
            changed.samples = list(raw.samples)
            path, target = changed.samples[0]
            changed.samples[0] = (path.replace("0.JPEG", "different.JPEG"), target)
            with self.assertRaises(ValueError):
                _load_manifest(changed, manifest_path, 0.05, 2026)

    def test_identical_pretrained_stem_and_new_head(self):
        config = ROOT / "weights/mcunet-256kb-1mb_imagenet.json"
        weights = ROOT / "weights/mcunet-256kb-1mb_imagenet.pth"
        reference_stem = reference_head = None
        feature_shapes = []
        for variant in VARIANT_SUBBANDS:
            model = load_model(config, weights, variant, num_classes=200).eval()
            stem = model.first_conv.conv.weight[:, :3]
            head = model.classifier.linear.weight
            if reference_stem is None:
                reference_stem, reference_head = stem.detach().clone(), head.detach().clone()
            self.assertTrue(torch.equal(stem, reference_stem))
            self.assertTrue(torch.equal(head, reference_head))
            self.assertEqual(torch.count_nonzero(model.first_conv.conv.weight[:, 3:]).item(), 0)
            sizes = []
            handles = [module.register_forward_hook(lambda m, i, o: sizes.append(tuple(o.shape[-2:])))
                       for module in [model.first_conv, *model.blocks]]
            resolution = 160 if variant == "rgb" else 80
            with torch.no_grad():
                output = model(torch.rand(1, model.first_conv.in_channels, resolution, resolution))
            self.assertEqual(tuple(output.shape), (1, 200))
            feature_shapes.append(sizes)
            for handle in handles:
                handle.remove()
        self.assertTrue(all(shapes == feature_shapes[0] for shapes in feature_shapes))


if __name__ == "__main__":
    unittest.main()
