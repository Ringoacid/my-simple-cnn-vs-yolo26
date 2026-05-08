import torch


class FlattenLayer:
    def forward(self, x):
        """
        順伝播

        Parameters:
        x (torch.Tensor): 入力テンソル (batch_size, channels, height, width)

        Returns:
        torch.Tensor: 出力ベクトル (batch_size, channels * height * width)
        """
        self._input_shape = x.shape
        batch_size = x.shape[0]
        # バッチ次元は保持したまま空間次元だけを展開するため、出力は (batch_size, features) の2次元テンソル（＝行列）になる
        return x.reshape(batch_size, -1)


    def backward(self, grad_output):
        """
        逆伝播

        Parameters:
        grad_output (torch.Tensor): 出力側からの勾配 (batch_size, channels * height * width)

        Returns:
        torch.Tensor: 入力に対する勾配 (batch_size, channels, height, width)
        """
        return grad_output.reshape(self._input_shape)


# --- 以下、テスト用のコード ---

def main():
    flatten = FlattenLayer()

    x = torch.randn(2, 3, 8, 8)

    # 順伝播テスト
    out = flatten.forward(x)
    print(f"入力形状: {x.shape}")
    print(f"出力形状: {out.shape}")

    # PyTorch autograd と逆伝播を比較
    x_ag = x.clone().requires_grad_(True)
    out_ag = x_ag.reshape(x_ag.shape[0], -1)
    grad_output = torch.randn_like(out_ag)
    out_ag.backward(grad_output)

    grad_input = flatten.backward(grad_output)
    diff = (grad_input - x_ag.grad).abs().max().item()
    print(f"逆伝播最大差: {diff:.2e} (ほぼ 0 なら正しい)")


if __name__ == "__main__":
    main()
