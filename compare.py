"""
自作 CNN モデルと YOLO26n のトレーニング結果を比較するスクリプト。

使用例:
    # 最新の実行結果を自動検出して比較
    conda run -n pt python compare.py

    # 実行ディレクトリを明示的に指定
    conda run -n pt python compare.py \\
        --custom runs/train/exp1 \\
        --yolo   runs/detect/train-2
"""

import argparse
import csv
from pathlib import Path

_ROOT = Path(__file__).parent


# --- 実行ディレクトリの自動検出 ---

def _latest_dir(base: Path) -> Path | None:
    """base 以下のサブディレクトリのうち、最も新しいものを返す。"""
    dirs = sorted(base.glob('*/'), key=lambda p: p.stat().st_mtime, reverse=True)
    return dirs[0] if dirs else None


def _find_custom_dir() -> Path | None:
    base = _ROOT / 'runs' / 'train'
    return _latest_dir(base) if base.exists() else None


def _find_yolo_dir() -> Path | None:
    base = _ROOT / 'runs' / 'detect'
    return _latest_dir(base) if base.exists() else None


# --- CSV 読み込み ---

def _load_csv(path: Path) -> list[dict]:
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def load_custom(run_dir: Path) -> list[dict]:
    """
    自作モデルの results.csv を読み込む。
    列: epoch, train_loss, val_loss, precision, recall, f1
    """
    path = run_dir / 'results.csv'
    rows = _load_csv(path)
    result = []
    for r in rows:
        result.append({
            'epoch':      int(r['epoch']),
            'train_loss': float(r['train_loss']),
            'val_loss':   float(r['val_loss']),
            'precision':  float(r['precision']),
            'recall':     float(r['recall']),
            'f1':         float(r['f1']),
        })
    return result


def load_yolo(run_dir: Path) -> list[dict]:
    """
    YOLO の results.csv を読み込む。
    train_loss = box_loss + cls_loss（dfl_loss は除外）
    val_loss   = val/box_loss + val/cls_loss
    """
    path = run_dir / 'results.csv'
    rows = _load_csv(path)
    result = []
    for r in rows:
        # 列名の前後スペースを除去
        r = {k.strip(): v.strip() for k, v in r.items()}
        train_loss = float(r['train/box_loss']) + float(r['train/cls_loss'])
        val_loss   = float(r['val/box_loss'])   + float(r['val/cls_loss'])
        result.append({
            'epoch':      int(r['epoch']),
            'train_loss': train_loss,
            'val_loss':   val_loss,
            'precision':  float(r['metrics/precision(B)']),
            'recall':     float(r['metrics/recall(B)']),
            'map50':      float(r['metrics/mAP50(B)']),
            'map50_95':   float(r['metrics/mAP50-95(B)']),
        })
    return result


# --- サマリーテーブル出力 ---

def print_summary(custom: list[dict], yolo: list[dict]) -> None:
    """最終エポックのメトリクスを並べて表示する。"""
    c = custom[-1]
    y = yolo[-1]

    print("=" * 62)
    print(f"{'指標':<20} {'自作 CNN':>18} {'YOLO26n':>18}")
    print("-" * 62)
    print(f"{'エポック数':<20} {c['epoch']:>18d} {y['epoch']:>18d}")
    print(f"{'train loss (最終)':<20} {c['train_loss']:>18.4f} {y['train_loss']:>18.4f}")
    print(f"{'val loss (最終)':<20} {c['val_loss']:>18.4f} {y['val_loss']:>18.4f}")
    print(f"{'Precision (最終)':<20} {c['precision']:>18.4f} {y['precision']:>18.4f}")
    print(f"{'Recall (最終)':<20} {c['recall']:>18.4f} {y['recall']:>18.4f}")
    print(f"{'F1 / mAP50 (最終)':<20} {c['f1']:>18.4f} {y['map50']:>18.4f}")
    print(f"{'mAP50-95 (最終)':<20} {'—':>18} {y['map50_95']:>18.4f}")
    print("=" * 62)
    print("  ※ YOLO の F1 列は mAP50 で代替しています。")
    print("  ※ 自作 CNN の損失は自作実装、YOLO は box_loss + cls_loss の合計。")
    print()


# --- グラフ描画 ---

def plot_comparison(custom: list[dict], yolo: list[dict], save_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib が見つかりません。グラフの保存をスキップします。")
        return

    save_dir.mkdir(parents=True, exist_ok=True)

    ce = [r['epoch'] for r in custom]
    ye = [r['epoch'] for r in yolo]

    panels = [
        ('Train Loss',  [r['train_loss'] for r in custom], [r['train_loss'] for r in yolo],  'Loss',      'train_loss.png'),
        ('Val Loss',    [r['val_loss']   for r in custom], [r['val_loss']   for r in yolo],  'Loss',      'val_loss.png'),
        ('Precision',   [r['precision']  for r in custom], [r['precision']  for r in yolo],  'Precision', 'precision.png'),
        ('Recall',      [r['recall']     for r in custom], [r['recall']     for r in yolo],  'Recall',    'recall.png'),
        ('F1 / mAP50',  [r['f1']         for r in custom], [r['map50']      for r in yolo],  'Score',     'f1_map50.png'),
    ]

    for title, c_vals, y_vals, ylabel, fname in panels:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(ce, c_vals, marker='o', label='Custom CNN', color='steelblue')
        ax.plot(ye, y_vals, marker='s', label='YOLO26n',    color='tomato')
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.4)
        fig.tight_layout()
        out = save_dir / fname
        fig.savefig(out, dpi=100)
        plt.close(fig)
        print(f"  保存: {out}")

    # 全パネルを 1 枚にまとめた概要図
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Custom CNN vs YOLO26n — Training Comparison', fontsize=14)
    axes_flat = axes.flatten()
    for idx, (title, c_vals, y_vals, ylabel, _) in enumerate(panels):
        ax = axes_flat[idx]
        ax.plot(ce, c_vals, marker='o', label='Custom CNN', color='steelblue')
        ax.plot(ye, y_vals, marker='s', label='YOLO26n',    color='tomato')
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.4)

    # 最後のパネルに mAP50-95 (YOLO のみ)
    ax = axes_flat[5]
    ax.plot(ye, [r['map50_95'] for r in yolo], marker='s', label='YOLO26n', color='tomato')
    ax.set_title('mAP50-95 (YOLO26n only)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('mAP50-95')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.4)

    fig.tight_layout()
    out = save_dir / 'comparison_overview.png'
    fig.savefig(out, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f"  概要図を保存: {out}")


# --- メイン ---

def main():
    parser = argparse.ArgumentParser(description='自作 CNN vs YOLO26n の比較')
    parser.add_argument('--custom', type=Path, default=None,
                        help='自作モデルの実行ディレクトリ (例: runs/train/exp1)')
    parser.add_argument('--yolo',   type=Path, default=None,
                        help='YOLO の実行ディレクトリ (例: runs/detect/train-2)')
    parser.add_argument('--out',    type=Path, default=_ROOT / 'runs' / 'compare',
                        help='グラフの出力先ディレクトリ (デフォルト: runs/compare)')
    args = parser.parse_args()

    # 実行ディレクトリの解決
    custom_dir = args.custom or _find_custom_dir()
    yolo_dir   = args.yolo   or _find_yolo_dir()

    if custom_dir is None:
        print("エラー: 自作モデルの実行ディレクトリが見つかりません。--custom で指定してください。")
        return
    if yolo_dir is None:
        print("エラー: YOLO の実行ディレクトリが見つかりません。--yolo で指定してください。")
        return

    print(f"自作 CNN: {custom_dir}")
    print(f"YOLO26n: {yolo_dir}")
    print()

    custom = load_custom(custom_dir)
    yolo   = load_yolo(yolo_dir)

    print_summary(custom, yolo)

    print("グラフを生成中...")
    plot_comparison(custom, yolo, args.out)
    print()
    print(f"完了。グラフは {args.out} に保存されました。")


if __name__ == '__main__':
    main()
