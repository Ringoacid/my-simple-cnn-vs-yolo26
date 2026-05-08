# my-simple-cnn-vs-yolo26 — CNNスクラッチ実装による顔検出

PyTorchの自動微分（autograd）を使わず、テンソル演算のみでCNNをゼロから実装し、顔検出タスクに適用するプロジェクトです。CPU・GPU（CUDA）の両方で動作します。

以下のURLから、カメラを使用して自作CNNとYOLO26nでリアルタイム顔検出を行うデモも公開しています。
カメラを使用しますが、データはクライアント側で処理されるため、安心してください。

https://my-simple-cnn-vs-yolo26.vercel.app/

## 特徴

- **im2col** を用いた畳み込み層の手実装（`layers/conv_layer.py`）
- forward / backward / update_parameters をすべて自前で実装
- YOLO形式のデータセットに対応したDataLoader
- Ultralytics YOLO26n との性能比較スクリプト

## プロジェクト構成

```
mycnn/
├── layers/              # レイヤー実装
│   ├── conv_layer.py        # 畳み込み層（im2col + 行列積）
│   ├── activation_layer.py  # ReLU
│   ├── pool_layer.py        # Max / Avg プーリング
│   ├── flatten_layer.py     # 平坦化
│   ├── fc_layer.py          # 全結合層
│   └── dropout_layer.py     # ドロップアウト
├── models/              # モデルコンポーネント
│   ├── backbone.py          # Conv→ReLU→Pool × 5段（stride=32）
│   └── detection_head.py    # 1×1 conv + sigmoid + 損失計算 + NMS
├── utils/               # ユーティリティ
│   ├── dataloader.py        # YOLO形式バッチ読み込み
│   └── img_to_tensor.py     # 単一画像のテンソル変換
├── dataset/             # YOLO形式顔検出データセット
│   └── data.yaml
├── train.py             # 自作CNNのトレーニングループ
├── train_yolo.py        # YOLO26n トレーニング（Ultralytics）
├── compare.py           # 結果の比較・グラフ出力
└── sample.jpg           # テスト用画像
```

## セットアップ

### conda 環境の作成

```bash
conda create -n pt python=3.13
conda activate pt
# CUDA 12.8 対応版（GPU 環境の場合）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
# CPU のみの場合
# pip install torch torchvision
pip install ultralytics matplotlib
```

## 使い方

### 自作CNNのトレーニング

```bash
# 動作確認（各 epoch 2 バッチのみ）
conda run -n pt python train.py --max-batches 2

# 本格トレーニング
conda run -n pt python train.py --epochs 10 --batch-size 4 --lr 1e-3
```

| オプション | デフォルト | 説明 |
|---|---|---|
| `--epochs` | 3 | エポック数 |
| `--batch-size` | 2 | バッチサイズ |
| `--lr` | 1e-3 | 学習率 |
| `--max-batches` | なし | 1 epoch あたりの最大バッチ数（動作確認用） |

結果は `runs/train/exp{N}/` に保存されます。

### YOLO26n トレーニング（比較用）

```bash
conda run -n pt python train_yolo.py
```

## アーキテクチャ

### バックボーン

```
入力 (batch, 3, 640, 640)
  ↓ Conv(3→8, 3×3) + ReLU + MaxPool(2×2)
  ↓ Conv(8→16, 3×3) + ReLU + MaxPool(2×2)
  ↓ Conv(16→32, 3×3) + ReLU + MaxPool(2×2)
  ↓ Conv(32→32, 3×3) + ReLU + MaxPool(2×2)
  ↓ Conv(32→32, 3×3) + ReLU + MaxPool(2×2)
出力 (batch, 32, 20, 20)  [stride = 32]
```

### 検出ヘッド

- 1×1 畳み込みで各グリッドセルから `(cx, cy, w, h, objectness, class)` を予測
- 損失: `L_total = L_cls + 5 × L_box + L_obj`（MSE + BCE）
- 推論時に NMS（Non-Maximum Suppression）を適用

## データセット

YOLO形式の顔検出データセット（クラス数: 1）

| split | 枚数 |
|---|---|
| train | 6,889 |
| valid | 1,966 |
| test | 986 |

## モデル比較結果（100エポック）

自作CNNとYOLO26nを同じデータセット・同じエポック数でトレーニングし、性能を比較しました。

| 指標 | 自作CNN | YOLO26n |
|---|---|---|
| Precision | 0.048 | **0.939** |
| Recall | 0.224 | **0.938** |
| F1 / mAP50 | 0.079 | **0.982** |
| mAP50-95 | — | **0.769** |
| val loss (最終) | 0.870 | **0.766** |

比較グラフは `runs/compare/` に保存されています（`python compare.py` で生成）。

![比較グラフ](runs/compare/comparison_overview.png)

Lossは、両モデルで算出方法やスケールが異なるため、直接比較は難しい点に注意してください。
YOLO26nは精度・再現率ともに自作CNNを大きく上回りました。自作CNNは損失が緩やかに低下し続けているものの、検出精度はまだ低く、アーキテクチャやアンカー設計の改善が課題です。

## ライセンス

[MIT License](LICENSE)
