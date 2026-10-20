"""Read-only checkpoint evaluation; run in the original JupyterLab project.

No training, downloads, or checkpoint modifications. Output goes to a new folder.
CKA uses all test examples; Procrustes is fitted on a fixed half and evaluated
on the other half. Both raw and row-normalized representations are measured.
"""
import argparse
import hashlib
import importlib.util
import itertools
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def cka(x, y):
    x = x - x.mean(0)
    y = y - y.mean(0)
    den = np.linalg.norm(x.T @ x) * np.linalg.norm(y.T @ y)
    return float(np.linalg.norm(x.T @ y) ** 2 / den) if den > 0 else None


def procrustes(x, y, fit, test):
    # No translation or rescaling: isolate orthogonal coordinate alignment.
    u, _, vt = np.linalg.svd(x[fit].T @ y[fit], full_matrices=False)
    q = u @ vt
    before = np.square(x[test] - y[test]).sum(1).mean()
    after = np.square(x[test] @ q - y[test]).sum(1).mean()
    return {"before_mse": float(before), "after_mse": float(after),
            "fraction_removed": float(1 - after / before) if before > 0 else None}


def self_test():
    rng = np.random.default_rng(71)
    x = rng.normal(size=(200, 8))
    q, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    y = x @ q
    assert abs(cka(x, y) - 1) < 1e-10
    p = procrustes(x, y, np.arange(100), np.arange(100, 200))
    assert p['after_mse'] < 1e-20
    assert abs(cka(x, x) - 1) < 1e-10
    assert cka(np.zeros_like(x), y) is None
    print('Metric self-tests passed.')


def save(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='/workspace/users/ashkezari/prj')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    self_test()
    if args.self_test:
        return
    import torch
    import torchvision
    from torch.utils.data import DataLoader
    from torchvision import transforms

    root = Path(args.root).resolve()
    source = root / 'sml/models_resnet18.py'
    spec = importlib.util.spec_from_file_location('audit_original_models', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    torch.manual_seed(20260929)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    output = root / 'results' / ('checkpoint_audit_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    output.mkdir(parents=True, exist_ok=False)
    files = sorted((root / 'checkpoints/exp6_resnet18').glob('*.pt'))
    if not files:
        raise FileNotFoundError('No ResNet-18 checkpoints found.')
    report = {'protocol': {'device': device, 'python': platform.python_version(),
        'torch': torch.__version__, 'torchvision': torchvision.__version__,
        'model_source': str(source), 'model_source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'batch_size': args.batch_size, 'dataset': 'CIFAR10 test, all 10000 samples',
        'normalization_mean': [0.4914, 0.4822, 0.4465],
        'normalization_std': [0.2470, 0.2435, 0.2616],
        'procrustes': 'orthogonal, no translation/scaling; fixed random 50/50 fit/evaluation split; seed 20260929',
        'cka': 'centered linear CKA in float64, all samples; raw and row-normalized features',
        'aggregation': 'sample-weighted; legacy equal-batch D_rep also provided',
        'checkpoint_count': len(files)}, 'health': [], 'evaluations': []}
    result_path = output / 'audit_results.json'
    # First audit every file, even if dataset loading later fails.
    for path in files:
        print('Health:', path.name, flush=True)
        record = {'file': path.name}
        try:
            # These are the user's own trusted training checkpoints.
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
            state = ckpt['trainer']['models']
            bad = [k for k, v in state.items() if torch.is_tensor(v)
                   and (v.is_floating_point() or v.is_complex()) and not torch.isfinite(v).all().item()]
            epochs = int(ckpt['trainer']['completed_epochs'])
            config = ckpt['config']
            record.update(config=config, completed_epochs=epochs, seed=ckpt.get('seed'),
                          data=ckpt.get('data'), nonfinite_model_tensors=bad,
                          status='nonfinite_weights' if bad else ('incomplete' if epochs < config['epochs'] else 'ready'))
            del state, ckpt
        except Exception as exc:
            record.update(status='load_error', error=f'{type(exc).__name__}: {exc}')
        report['health'].append(record)
        save(result_path, report)

    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(
        report['protocol']['normalization_mean'], report['protocol']['normalization_std'])])
    dataset = torchvision.datasets.CIFAR10(root=str(root / 'data'), train=False, download=False, transform=tf)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    order = np.random.default_rng(20260929).permutation(len(dataset))
    fit, test = np.split(order, [len(order) // 2])
    for health in report['health']:
        record = {'file': health['file'], 'status': health['status']}
        if health['status'] != 'ready':
            report['evaluations'].append(record)
            save(result_path, report)
            continue
        print('Evaluate:', health['file'], flush=True)
        models = None
        try:
            c = health['config']
            models = module.build_cohort(K=c['K'], num_classes=c['num_classes'],
                        repr_dim=c['repr_dim'], widths=c.get('widths'), backbone=c['backbone'])
            ckpt = torch.load(root / 'checkpoints/exp6_resnet18' / health['file'], map_location='cpu', weights_only=False)
            models.load_state_dict(ckpt['trainer']['models'], strict=True)
            del ckpt
            models.to(device).eval()
            hs, zs, labels, batch_sizes = [[] for _ in models], [[] for _ in models], [], []
            with torch.inference_mode():
                for x, y in loader:
                    x = x.to(device)
                    for k, model in enumerate(models):
                        h, z = model(x)
                        if not torch.isfinite(h).all() or not torch.isfinite(z).all():
                            raise ValueError('Nonfinite features/logits during inference')
                        hs[k].append(h.cpu().numpy())
                        zs[k].append(z.cpu().numpy())
                    labels.append(y.numpy())
                    batch_sizes.append(len(y))
            h = [np.concatenate(a).astype(np.float64) for a in hs]
            z = [np.concatenate(a).astype(np.float64) for a in zs]
            y = np.concatenate(labels)
            norms = [np.linalg.norm(a, axis=1) for a in h]
            if any(np.any(a == 0) for a in norms):
                raise ValueError('Zero feature norm; spherical geometry undefined')
            r = [a / n[:, None] for a, n in zip(h, norms)]
            probs = []
            for a in z:
                e = np.exp(a - a.max(1, keepdims=True)); probs.append(e / e.sum(1, keepdims=True))
            pred = [a.argmax(1) for a in probs]
            ensemble = np.mean(probs, axis=0)
            pair_results, distances = [], []
            for i, j in itertools.combinations(range(len(h)), 2):
                s = np.square(r[i] - r[j]).sum(1); distances.append(s)
                pair_results.append({'peers': [i, j], 'D_rep': float(s.mean()),
                    'E_d0': float(np.square(s - c['d0'] ** 2).mean()),
                    'linear_cka_raw': cka(h[i], h[j]), 'linear_cka_normalized': cka(r[i], r[j]),
                    'procrustes_raw': procrustes(h[i], h[j], fit, test),
                    'procrustes_normalized': procrustes(r[i], r[j], fit, test),
                    'prediction_disagreement': float(np.mean(pred[i] != pred[j]))})
            ds = np.mean(distances, axis=0)
            boundaries = np.cumsum([0] + batch_sizes)
            record.update(status='ok', n_samples=len(y),
                ensemble_acc=float(np.mean(ensemble.argmax(1) == y)),
                per_model_acc=[float(np.mean(p == y)) for p in pred],
                D_rep=float(ds.mean()), legacy_equal_batch_D_rep=float(np.mean([
                    ds[a:b].mean() for a, b in zip(boundaries[:-1], boundaries[1:])])),
                min_feature_norm=[float(n.min()) for n in norms], pairs=pair_results)
        except Exception as exc:
            record.update(status='evaluation_error', error=f'{type(exc).__name__}: {exc}')
        finally:
            del models
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        report['evaluations'].append(record)
        save(result_path, report)
        print(' ->', record['status'], record.get('ensemble_acc', ''), flush=True)
    print('\nFinished. Download only:', result_path, flush=True)
    print('Failures remain in the report; no failed run is silently discarded.')


if __name__ == '__main__':
    main()
