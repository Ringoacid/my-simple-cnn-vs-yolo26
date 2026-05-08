"""
顔検出モデルのトレーニングスクリプト。

注意: 各 ConvLayer は im2col（純 Python + PyTorch CPU）で実装されているため、
640×640 のフルスケールで学習させると 1 エポックに数時間〜数十時間かかる場合があります。
動作確認には --max-batches オプションで実行バッチ数を制限することを推奨します。

使用例:
    # 動作確認（各 split 2 バッチのみ）
    conda run -n pt python train.py --max-batches 2

    # 本格的なトレーニング
    conda run -n pt python train.py --epochs 10 --batch-size 4 --lr 1e-3
"""

import argparse
import csv
import time
from pathlib import Path

import torch

from layers import ConvLayer
from models import Backbone, DetectionHead, nms
from utils import get_dataloader

_ROOT = Path(__file__).parent


def _next_run_dir():
    base = _ROOT / 'runs' / 'train'
    i = 1
    while True:
        d = base / f'exp{i}'
        if not d.exists():
            return d
        i += 1


# --- チェックポイント ---

def _backbone_state(backbone):
    sd = {}
    conv_idx = 0
    for layer in backbone.layers:
        if isinstance(layer, ConvLayer):
            sd[f'conv{conv_idx}.weights'] = layer.weights.clone()
            if layer.bias is not None:
                sd[f'conv{conv_idx}.bias'] = layer.bias.clone()
            conv_idx += 1
    return sd


def _backbone_load(backbone, sd):
    conv_idx = 0
    for layer in backbone.layers:
        if isinstance(layer, ConvLayer):
            layer.weights = sd[f'conv{conv_idx}.weights']
            if layer.bias is not None and f'conv{conv_idx}.bias' in sd:
                layer.bias = sd[f'conv{conv_idx}.bias']
            conv_idx += 1


def _head_state(head):
    sd = {'conv.weights': head.conv.weights.clone()}
    if head.conv.bias is not None:
        sd['conv.bias'] = head.conv.bias.clone()
    return sd


def _head_load(head, sd):
    head.conv.weights = sd['conv.weights']
    if head.conv.bias is not None and 'conv.bias' in sd:
        head.conv.bias = sd['conv.bias']


def save_checkpoint(backbone, head, path):
    torch.save({'backbone': _backbone_state(backbone), 'head': _head_state(head)}, path)


def load_checkpoint(backbone, head, path):
    ckpt = torch.load(path, map_location='cpu')
    _backbone_load(backbone, ckpt['backbone'])
    _head_load(head, ckpt['head'])


# --- IoU ---

def _box_iou(boxes1, boxes2):
    """boxes1: (M, 4) cx,cy,w,h  boxes2: (N, 4) → (M, N) IoU"""
    b1_x1 = boxes1[:, 0:1] - boxes1[:, 2:3] / 2
    b1_y1 = boxes1[:, 1:2] - boxes1[:, 3:4] / 2
    b1_x2 = boxes1[:, 0:1] + boxes1[:, 2:3] / 2
    b1_y2 = boxes1[:, 1:2] + boxes1[:, 3:4] / 2

    b2_x1 = (boxes2[:, 0] - boxes2[:, 2] / 2).unsqueeze(0)
    b2_y1 = (boxes2[:, 1] - boxes2[:, 3] / 2).unsqueeze(0)
    b2_x2 = (boxes2[:, 0] + boxes2[:, 2] / 2).unsqueeze(0)
    b2_y2 = (boxes2[:, 1] + boxes2[:, 3] / 2).unsqueeze(0)

    inter = (torch.clamp(torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1), min=0)
             * torch.clamp(torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1), min=0))
    area1 = boxes1[:, 2:3] * boxes1[:, 3:4]
    area2 = (boxes2[:, 2] * boxes2[:, 3]).unsqueeze(0)
    return inter / (area1 + area2 - inter + 1e-7)


# --- PR / F1 データ計算 ---

def compute_pr_data(backbone, head, loader, max_batches=None, device=None, iou_threshold=0.5):
    """
    検証セット全体で予測を収集し、PR・F1 カーブ用のデータを返す。

    Returns:
    dict: precisions / recalls / f1s / thresholds / best_f1 / best_P / best_R
    """
    all_entries = []  # (conf, is_tp)
    total_gt = 0

    for i, (images, targets) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        if device is not None:
            images = images.to(device)

        feat = backbone.forward(images)
        pred = head.forward(feat)

        for b in range(images.shape[0]):
            gt = targets[b]
            if gt is None or gt.numel() == 0:
                gt_boxes = torch.zeros(0, 4, device=pred.device)
            else:
                gt_boxes = gt[:, 1:].to(pred.device)  # (N, 4) cx,cy,w,h
            total_gt += gt_boxes.shape[0]

            cx, cy, w, h, obj, cls = head.decode(pred[b:b + 1])
            conf_flat = (obj[0] * cls[0, 0]).flatten()
            boxes_flat = torch.stack(
                [cx[0].flatten(), cy[0].flatten(), w[0].flatten(), h[0].flatten()], dim=1
            )

            kept = nms(boxes_flat, conf_flat, iou_threshold=0.4, score_threshold=0.01)
            if not kept:
                continue

            pb = boxes_flat[kept]
            pc = conf_flat[kept]
            order = pc.argsort(descending=True)
            pb, pc = pb[order], pc[order]

            gt_matched = torch.zeros(gt_boxes.shape[0], dtype=torch.bool, device=pred.device)
            for j in range(pb.shape[0]):
                score = pc[j].item()
                if gt_boxes.shape[0] == 0:
                    all_entries.append((score, False))
                    continue
                iou = _box_iou(pb[j:j + 1], gt_boxes)[0]
                best_iou, best_idx = iou.max(dim=0)
                if best_iou.item() >= iou_threshold and not gt_matched[best_idx.item()]:
                    gt_matched[best_idx.item()] = True
                    all_entries.append((score, True))
                else:
                    all_entries.append((score, False))

    empty = {'precisions': [], 'recalls': [], 'f1s': [], 'thresholds': [],
             'best_f1': 0.0, 'best_P': 0.0, 'best_R': 0.0}
    if not all_entries or total_gt == 0:
        return empty

    all_entries.sort(key=lambda x: -x[0])
    thresholds = [e[0] for e in all_entries]
    is_tp = torch.tensor([e[1] for e in all_entries], dtype=torch.float32)

    tp_cum = is_tp.cumsum(0)
    fp_cum = (1 - is_tp).cumsum(0)
    precisions = (tp_cum / (tp_cum + fp_cum + 1e-7)).tolist()
    recalls    = (tp_cum / (total_gt + 1e-7)).tolist()
    f1s        = [2 * p * r / (p + r + 1e-7) for p, r in zip(precisions, recalls)]

    best_idx = int(torch.tensor(f1s).argmax().item())
    return {
        'precisions': precisions,
        'recalls':    recalls,
        'f1s':        f1s,
        'thresholds': thresholds,
        'best_f1':    f1s[best_idx],
        'best_P':     precisions[best_idx],
        'best_R':     recalls[best_idx],
    }


# --- カーブ保存 ---

def save_curves(pr_data, save_dir):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib が見つかりません。カーブの保存をスキップします。")
        return

    save_dir = Path(save_dir)
    P = pr_data['precisions']
    R = pr_data['recalls']
    F = pr_data['f1s']
    T = pr_data['thresholds']
    if not P:
        print("  予測データなし。カーブの保存をスキップします。")
        return

    # PR カーブ
    fig, ax = plt.subplots()
    ax.plot(R, P, color='blue')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('Precision-Recall Curve')
    ax.grid(True)
    fig.savefig(save_dir / 'PR_curve.png', dpi=100, bbox_inches='tight')
    plt.close(fig)

    # F1 カーブ（横軸: confidence threshold、左が高信頼・右が低信頼）
    fig, ax = plt.subplots()
    ax.plot(T, F, color='green')
    ax.set_xlabel('Confidence Threshold')
    ax.set_ylabel('F1')
    ax.set_xlim(max(T + [1.0]), min(T + [0.0]))  # 降順（高 conf → 低 conf）
    ax.set_ylim(0, 1)
    ax.set_title(f'F1-Confidence Curve  (best F1={pr_data["best_f1"]:.3f})')
    ax.grid(True)
    fig.savefig(save_dir / 'F1_curve.png', dpi=100, bbox_inches='tight')
    plt.close(fig)

    print(f"  カーブを保存: {save_dir / 'PR_curve.png'}, {save_dir / 'F1_curve.png'}")


# --- トレーニングステップ ---

def train_step(backbone, head, images, targets, lr):
    """
    1 バッチ分の forward → 損失計算 → backward → パラメータ更新を行う。

    Returns:
    float: このバッチのスカラー損失値
    """
    feat = backbone.forward(images)
    pred = head.forward(feat)
    loss, grad_pred = head.compute_loss(pred, targets)

    grad_feat, gw, gb = head.backward(feat, pred, grad_pred)
    backbone.backward(grad_feat)

    head.update_parameters(gw, gb, lr)
    backbone.update_parameters(lr)

    return loss.item()


def validate(backbone, head, loader, max_batches=None, device=None):
    """
    validation set の平均損失を計算する（パラメータ更新なし）。

    Returns:
    float: 平均損失値
    """
    total_loss = 0.0
    count = 0
    for i, (images, targets) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        if device is not None:
            images = images.to(device)
        feat = backbone.forward(images)
        pred = head.forward(feat)
        loss, _ = head.compute_loss(pred, targets)
        total_loss += loss.item()
        count += 1
    return total_loss / count if count > 0 else float('nan')


# --- メインループ ---

def train(epochs, batch_size, lr, max_batches, log_interval):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    backbone = Backbone(out_channels=32, device=device)
    head = DetectionHead(in_channels=32, num_classes=1, device=device)

    run_dir = _next_run_dir()
    weights_dir = run_dir / 'weights'
    weights_dir.mkdir(parents=True)

    results_path = run_dir / 'results.csv'
    csv_file = open(results_path, 'w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['epoch', 'train_loss', 'val_loss', 'precision', 'recall', 'f1'])

    train_loader = get_dataloader('train', batch_size=batch_size, shuffle=True)
    valid_loader = get_dataloader('valid', batch_size=batch_size, shuffle=False)

    print(f"デバイス: {device}")
    print(f"結果の保存先: {run_dir}")
    print(f"トレーニング開始: epochs={epochs}, batch_size={batch_size}, lr={lr}")
    if max_batches is not None:
        print(f"  (--max-batches {max_batches}: 各 epoch で最大 {max_batches} バッチのみ実行)")
    print()

    best_val_loss = float('inf')
    last_pr_data = None

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        running_loss = 0.0
        n_batches = 0

        for i, (images, targets) in enumerate(train_loader):
            if max_batches is not None and i >= max_batches:
                break

            images = images.to(device)
            batch_loss = train_step(backbone, head, images, targets, lr)
            running_loss += batch_loss
            n_batches += 1

            if (i + 1) % log_interval == 0:
                avg = running_loss / n_batches
                elapsed = time.time() - epoch_start
                print(f"  epoch {epoch}/{epochs}  batch {i+1}  "
                      f"loss={avg:.4f}  経過 {elapsed:.1f}s")

        train_loss = running_loss / n_batches if n_batches > 0 else float('nan')

        val_max = max_batches
        valid_loss = validate(backbone, head, valid_loader, max_batches=val_max, device=device)

        last_pr_data = compute_pr_data(
            backbone, head, valid_loader, max_batches=val_max, device=device)
        P  = last_pr_data['best_P']
        R  = last_pr_data['best_R']
        F1 = last_pr_data['best_f1']

        elapsed = time.time() - epoch_start
        print(f"[epoch {epoch}/{epochs}]  "
              f"train_loss={train_loss:.4f}  valid_loss={valid_loss:.4f}  "
              f"P={P:.4f}  R={R:.4f}  F1={F1:.4f}  時間={elapsed:.1f}s")

        csv_writer.writerow([epoch,
                             f'{train_loss:.6f}', f'{valid_loss:.6f}',
                             f'{P:.6f}', f'{R:.6f}', f'{F1:.6f}'])
        csv_file.flush()

        save_checkpoint(backbone, head, weights_dir / 'last.pt')
        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            save_checkpoint(backbone, head, weights_dir / 'best.pt')
            print(f"  → best.pt を更新 (valid_loss={valid_loss:.4f})")

        print()

    csv_file.close()

    print("PR / F1 カーブを保存中...")
    if last_pr_data is not None:
        save_curves(last_pr_data, run_dir)

    print(f"\nトレーニング完了。結果を {run_dir} に保存しました。")
    print(f"  weights/best.pt  (best valid_loss={best_val_loss:.4f})")
    print(f"  weights/last.pt")
    print(f"  results.csv")


def main():
    parser = argparse.ArgumentParser(description='顔検出 CNN のトレーニング')
    parser.add_argument('--epochs',      type=int,   default=3,    help='エポック数 (デフォルト: 3)')
    parser.add_argument('--batch-size',  type=int,   default=2,    help='バッチサイズ (デフォルト: 2)')
    parser.add_argument('--lr',          type=float, default=1e-3, help='学習率 (デフォルト: 1e-3)')
    parser.add_argument('--max-batches', type=int,   default=None,
                        help='1 epoch あたりの最大バッチ数（動作確認用）')
    parser.add_argument('--log-interval', type=int,  default=10,
                        help='損失をログする間隔（バッチ数）(デフォルト: 10)')
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_batches=args.max_batches,
        log_interval=args.log_interval,
    )


if __name__ == '__main__':
    main()
