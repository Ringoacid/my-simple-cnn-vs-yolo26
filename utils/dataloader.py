import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

DATASET_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset')

_to_tensor = transforms.ToTensor()  # PIL → (C, H, W) float32, [0, 1]


class FaceDataset(Dataset):
    """
    YOLO形式の顔検出データセット。
    画像は (3, 640, 640) の float32 テンソル（値域 [0, 1]）。
    ラベルは (N, 5) の float32 テンソル（各行が class_id, cx, cy, w, h）。
    """

    def __init__(self, split='train'):
        """
        Parameters:
        split (str): 'train' / 'valid' / 'test'
        """
        assert split in ('train', 'valid', 'test'), f"split は 'train'/'valid'/'test' のいずれかです: {split}"
        self.img_dir   = os.path.join(DATASET_ROOT, split, 'images')
        self.label_dir = os.path.join(DATASET_ROOT, split, 'labels')

        # 画像ファイルのパス一覧を収集
        exts = ('.jpg', '.jpeg', '.png')
        self.img_paths = sorted(
            os.path.join(self.img_dir, f)
            for f in os.listdir(self.img_dir)
            if f.lower().endswith(exts)
        )

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]

        # 画像を読み込んでテンソルに変換
        image = _to_tensor(Image.open(img_path).convert('RGB'))

        # 対応するラベルファイルのパスを組み立てる
        basename = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(self.label_dir, basename + '.txt')

        labels = _load_labels(label_path)
        return image, labels


def _load_labels(label_path):
    """
    YOLO形式のラベルファイルを読み込む。

    Returns:
    torch.Tensor: (N, 5) の float32 テンソル（class_id, cx, cy, w, h）。
                  ラベルなし（背景画像）の場合は shape (0, 5) を返す。
    """
    if not os.path.exists(label_path):
        return torch.zeros(0, 5)

    rows = []
    with open(label_path) as f:
        for line in f:
            line = line.strip()
            if line:
                values = list(map(float, line.split()))
                rows.append(values)

    if not rows:
        return torch.zeros(0, 5)

    return torch.tensor(rows, dtype=torch.float32)


def collate_fn(batch):
    """
    バッチ内でラベルの物体数が異なるため、カスタム collate が必要。

    Returns:
    tuple:
        images  (torch.Tensor): (batch, 3, 640, 640)
        targets (list[torch.Tensor]): 各要素が (N_i, 5)。
                 DetectionHead.compute_loss() にそのまま渡せる形式。
    """
    images, targets = zip(*batch)
    return torch.stack(images, dim=0), list(targets)


def get_dataloader(split='train', batch_size=8, shuffle=None, num_workers=0):
    """
    指定した split の DataLoader を返す。

    Parameters:
    split       (str): 'train' / 'valid' / 'test'
    batch_size  (int): バッチサイズ
    shuffle     (bool | None): None の場合、train のみ True にする
    num_workers (int): データ読み込みのワーカー数

    Returns:
    torch.utils.data.DataLoader
    """
    if shuffle is None:
        shuffle = (split == 'train')
    dataset = FaceDataset(split=split)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )


# --- 以下、テスト用のコード ---

def main():
    for split in ('train', 'valid', 'test'):
        ds = FaceDataset(split=split)
        print(f"[{split}] 画像数: {len(ds)}")

    # DataLoader の動作確認
    loader = get_dataloader(split='train', batch_size=4, shuffle=False)
    images, targets = next(iter(loader))

    print(f"\nバッチ確認:")
    print(f"  images 形状 : {images.shape}  (期待値: [4, 3, 640, 640])")
    print(f"  images 値域 : [{images.min():.3f}, {images.max():.3f}]  (期待値: [0, 1])")
    print(f"  targets 長さ: {len(targets)}  (期待値: 4)")

    for i, t in enumerate(targets):
        print(f"  targets[{i}] 形状: {t.shape}  (N, 5)")

    # DetectionHead.compute_loss が期待するフォーマットの確認
    # targets の各要素が (class_id, cx, cy, w, h) になっているか
    for i, t in enumerate(targets):
        if t.numel() > 0:
            print(f"\n  targets[{i}] の先頭行: class_id={t[0,0]:.0f}  cx={t[0,1]:.4f}  cy={t[0,2]:.4f}  w={t[0,3]:.4f}  h={t[0,4]:.4f}")
            break


if __name__ == "__main__":
    main()
