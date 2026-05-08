import torch
from layers import ConvLayer, ActivationLayer, PoolLayer


class Backbone:
    """
    5段の Conv→ReLU→MaxPool からなる特徴抽出バックボーン。
    入力 (batch, 3, 640, 640) → 出力 (batch, out_channels, 20, 20)。
    stride=32 は MaxPool×5 で実現する。
    """

    def __init__(self, out_channels=32, device=None):
        self.layers = [
            ConvLayer(3,            8,           3, padding=1, device=device), ActivationLayer('relu'), PoolLayer(2, 2, 'max'),
            ConvLayer(8,            16,          3, padding=1, device=device), ActivationLayer('relu'), PoolLayer(2, 2, 'max'),
            ConvLayer(16,           out_channels, 3, padding=1, device=device), ActivationLayer('relu'), PoolLayer(2, 2, 'max'),
            ConvLayer(out_channels, out_channels, 3, padding=1, device=device), ActivationLayer('relu'), PoolLayer(2, 2, 'max'),
            ConvLayer(out_channels, out_channels, 3, padding=1, device=device), ActivationLayer('relu'), PoolLayer(2, 2, 'max'),
        ]
        self._activations = []
        self._grads = []

    def forward(self, x):
        """
        Parameters:
        x (torch.Tensor): 入力画像 (batch, 3, H, W)

        Returns:
        torch.Tensor: 特徴マップ (batch, out_channels, H/32, W/32)
        """
        self._activations = [x]
        for layer in self.layers:
            x = layer.forward(x)
            self._activations.append(x)
        return self._activations[-1]

    def backward(self, grad):
        """
        各層に勾配を逆伝播し、ConvLayer の勾配を内部に保存する。

        Parameters:
        grad (torch.Tensor): DetectionHead からの勾配 (batch, out_channels, H/32, W/32)

        Returns:
        torch.Tensor: 入力画像への勾配（デバッグ用）
        """
        self._grads = []
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            x_in = self._activations[i]
            if isinstance(layer, ConvLayer):
                grad, gw, gb = layer.backward(x_in, grad)
                self._grads.append((layer, gw, gb))
            elif isinstance(layer, ActivationLayer):
                grad = layer.backward(x_in, grad)
            elif isinstance(layer, PoolLayer):
                grad = layer.backward(x_in, grad)
        return grad

    def update_parameters(self, lr):
        """
        backward() で蓄積した勾配で全 ConvLayer のパラメータを更新する（SGD）。

        Parameters:
        lr (float): 学習率
        """
        for layer, gw, gb in self._grads:
            layer.update_parameters(gw, gb, lr)


# --- 以下、テスト用のコード ---

def main():
    import math

    backbone = Backbone(out_channels=32)
    print(f"層数: {len(backbone.layers)}")

    # 順伝播テスト
    batch = 2
    x = torch.randn(batch, 3, 640, 640)
    feat = backbone.forward(x)
    print(f"\n入力形状: {x.shape}")
    print(f"特徴マップ形状: {feat.shape}  (期待値: [{batch}, 32, 20, 20])")

    # 逆伝播テスト（勾配が形状を保持するか確認）
    grad_feat = torch.randn_like(feat)
    grad_input = backbone.backward(grad_feat)
    print(f"\n入力への勾配形状: {grad_input.shape}  (入力と同じなら正しい)")
    print(f"ConvLayer 勾配の数: {len(backbone._grads)}  (期待値: 5)")

    # パラメータ更新テスト
    lr = 1e-3
    w_before = backbone.layers[0].weights.clone()
    backbone.update_parameters(lr)
    w_after = backbone.layers[0].weights
    changed = not torch.allclose(w_before, w_after)
    print(f"\nパラメータ更新確認: {'OK (変化あり)' if changed else 'NG (変化なし)'}")


if __name__ == "__main__":
    main()
