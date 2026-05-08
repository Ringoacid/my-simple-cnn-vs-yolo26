# プロジェクトの概要
[記事.md](記事.md)に示している通り、CNNの仕組みを解説するためのプロジェクト。畳み込み層などのCNNの構成要素を実装し、画像認識のタスクに適用することを目的としています。

## プロジェクトの構成

### `layers/` — レイヤー実装（forward / backward / update_parameters）
- `conv_layer.py`: 畳み込み層（im2col + 行列積）
- `activation_layer.py`: 活性化層（ReLU）
- `pool_layer.py`: プーリング層（Max / Avg）
- `flatten_layer.py`: 平坦化層
- `fc_layer.py`: 全結合層
- `dropout_layer.py`: ドロップアウト層

### `models/` — モデルコンポーネント
- `backbone.py`: バックボーン（Conv→ReLU→Pool × 5段、stride=32）。入力 `(batch, 3, 640, 640)` → 出力 `(batch, 32, 20, 20)`
- `detection_head.py`: 検出ヘッド（1×1 conv + sigmoid + 損失計算 + NMS）

### `utils/` — ユーティリティ
- `dataloader.py`: YOLO形式のラベルと画像をバッチ読み込みする DataLoader
- `img_to_tensor.py`: 単一画像をテンソルに変換するユーティリティ

### ルート
- `train.py`: トレーニングループ（forward → compute_loss → backward → update_parameters）
- `train_yolo.py`: Ultralyticsの最新モデル、YOLO26nでトレーニングする
- `dataset/`: YOLO形式の顔検出データセット（train: 6889枚 / valid: 1966枚 / test: 986枚、クラス数 1）
- `sample.jpg`: テスト用画像（640×640、左上から白・赤・緑・青・黒、残りは白）
- `記事.md`: CNNの仕組みを解説する記事

## 実行環境
Pythonコードの実行には conda の `pt` 環境を使用する。
```bash
conda run -n pt python ...
```
