"""
ONNX エクスポートスクリプト

カスタムCNNとYOLO26nをONNX形式にエクスポートし、
web/models/ に保存する。manifest.json も生成する。

実行:
    conda run -n pt python export_onnx.py
"""

import json
import shutil
from pathlib import Path

import torch
import torch.nn as nn
import onnx
from onnx import shape_inference

_ROOT = Path(__file__).parent
_WEB_MODELS = _ROOT / 'web' / 'models'


# ─────────────────────────────────────────────
# カスタム CNN の nn.Module ラッパー
# ─────────────────────────────────────────────

class CustomCNNModule(nn.Module):
    """
    手書き ConvLayer を nn.Module に変換したラッパー。
    チェックポイントの weights/bias を nn.Conv2d にコピーして ONNX エクスポートに使う。

    アーキテクチャ (backbone.py より):
      Block1: Conv(3→8,  3x3, pad=1) → ReLU → MaxPool(2)
      Block2: Conv(8→16, 3x3, pad=1) → ReLU → MaxPool(2)
      Block3: Conv(16→32,3x3, pad=1) → ReLU → MaxPool(2)
      Block4: Conv(32→32,3x3, pad=1) → ReLU → MaxPool(2)
      Block5: Conv(32→32,3x3, pad=1) → ReLU → MaxPool(2)
      Head:   Conv(32→6, 1x1)        → Sigmoid
    """

    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3,  8,  3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(8,  16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
        )
        self.head_conv = nn.Conv2d(32, 6, 1)

    def forward(self, x):
        return torch.sigmoid(self.head_conv(self.backbone(x)))


def load_custom_cnn(ckpt_path: Path) -> CustomCNNModule:
    """
    チェックポイントから重みをロードして CustomCNNModule を返す。

    backbone.layers のレイアウト: [Conv, ReLU, Pool, Conv, ReLU, Pool, ...]
    ckpt key    → nn.Sequential インデックス
    conv0.*     → backbone[0]   (Conv2d 3→8)
    conv1.*     → backbone[3]   (Conv2d 8→16)
    conv2.*     → backbone[6]   (Conv2d 16→32)
    conv3.*     → backbone[9]   (Conv2d 32→32)
    conv4.*     → backbone[12]  (Conv2d 32→32)
    head.conv.* → head_conv     (Conv2d 32→6)
    """
    ckpt = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
    model = CustomCNNModule()

    # backbone: conv0〜conv4 → nn.Sequential インデックス 0,3,6,9,12
    backbone_conv_indices = [0, 3, 6, 9, 12]
    backbone_sd = ckpt['backbone']
    for i, layer_idx in enumerate(backbone_conv_indices):
        layer = model.backbone[layer_idx]
        with torch.no_grad():
            layer.weight.data.copy_(backbone_sd[f'conv{i}.weights'])
            if f'conv{i}.bias' in backbone_sd:
                layer.bias.data.copy_(backbone_sd[f'conv{i}.bias'])

    # head
    head_sd = ckpt['head']
    with torch.no_grad():
        model.head_conv.weight.data.copy_(head_sd['conv.weights'])
        if 'conv.bias' in head_sd:
            model.head_conv.bias.data.copy_(head_sd['conv.bias'])

    model.eval()
    return model


def export_custom_cnn(out_path: Path) -> dict:
    ckpt_path = _ROOT / 'runs' / 'train' / 'latest' / 'weights' / 'best.pt'
    print(f'カスタムCNN チェックポイント: {ckpt_path}')
    model = load_custom_cnn(ckpt_path)

    dummy = torch.zeros(1, 3, 640, 640)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=['images'],
        output_names=['output'],
        opset_version=17,
    )
    print(f'カスタムCNN ONNX 保存: {out_path}')

    onnx_model = onnx.load(str(out_path))
    inferred = shape_inference.infer_shapes(onnx_model)
    output_shape = [d.dim_value for d in
                    inferred.graph.output[0].type.tensor_type.shape.dim]
    print(f'  出力形状: {output_shape}')
    return {'output_shape': output_shape}


def export_yolo26n(out_path: Path) -> dict:
    from ultralytics import YOLO
    ckpt_path = _ROOT / 'runs' / 'detect' / 'train-3' / 'weights' / 'best.pt'
    print(f'YOLO26n チェックポイント: {ckpt_path}')

    model = YOLO(str(ckpt_path))
    exported = model.export(
        format='onnx',
        imgsz=640,
        simplify=True,
        opset=17,
        half=False,
    )
    yolo_onnx_src = Path(str(exported))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(yolo_onnx_src), str(out_path))
    print(f'YOLO26n ONNX 保存: {out_path}')

    onnx_model = onnx.load(str(out_path))
    inferred = shape_inference.infer_shapes(onnx_model)
    output_shape = [d.dim_value for d in
                    inferred.graph.output[0].type.tensor_type.shape.dim]
    print(f'  出力形状: {output_shape}')
    return {'output_shape': output_shape}


def write_manifest(cnn_info: dict, yolo_info: dict):
    manifest = {
        'custom_cnn': {
            'file': 'custom_cnn.onnx',
            'input_shape': [1, 3, 640, 640],
            'output_shape': cnn_info['output_shape'],
            'format': 'cnn_grid',
            'grid_size': 20,
        },
        'yolo26n': {
            'file': 'yolo26n.onnx',
            'input_shape': [1, 3, 640, 640],
            'output_shape': yolo_info['output_shape'],
            'format': 'yolo_anchors',
        },
    }
    manifest_path = _WEB_MODELS / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f'manifest.json 保存: {manifest_path}')


if __name__ == '__main__':
    cnn_info  = export_custom_cnn(_WEB_MODELS / 'custom_cnn.onnx')
    yolo_info = export_yolo26n(_WEB_MODELS / 'yolo26n.onnx')
    write_manifest(cnn_info, yolo_info)
    print('\nエクスポート完了。')
