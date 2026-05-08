import torch
import math


class FCLayer:
    def __init__(self, in_features, out_features, bias=True):
        """
        全結合層の初期化

        Parameters:
        in_features  (int): 入力の特徴量数
        out_features (int): 出力の特徴量数
        bias         (bool): バイアスを使うか否か（デフォルトは True）
        """
        self.in_features = in_features
        self.out_features = out_features

        self.weights = self._he_initialization()

        self.use_bias = bias
        if self.use_bias:
            self.bias = self._bias_initialization()
        else:
            self.bias = None


    def _he_initialization(self):
        """
        Heの初期化（Kaiming Normal）による重みテンソルの生成
        """
        fan_in = self.in_features
        sigma = math.sqrt(2.0 / fan_in)
        return torch.randn(self.out_features, self.in_features) * sigma


    def _bias_initialization(self):
        """
        バイアスの初期化
        """
        bound = 1.0 / math.sqrt(self.in_features)
        return (torch.rand(self.out_features) * 2 - 1) * bound


    def forward(self, x):
        """
        順伝播

        Parameters:
        x (torch.Tensor): 入力テンソル (batch_size, in_features)

        Returns:
        torch.Tensor: 出力テンソル (batch_size, out_features)
        """
        # (batch, in_features) @ (in_features, out_features) → (batch, out_features)
        out = x @ self.weights.t()
        if self.use_bias:
            out = out + self.bias
        return out


    def backward(self, x, grad_output):
        """
        逆伝播

        Parameters:
        x           (torch.Tensor): 順伝播時の入力テンソル (batch_size, in_features)
        grad_output (torch.Tensor): 出力側からの勾配 (batch_size, out_features)

        Returns:
        tuple: (grad_input, grad_weights, grad_bias)
            grad_input  : 入力に対する勾配 (batch_size, in_features)
            grad_weights: 重みに対する勾配 (out_features, in_features)
            grad_bias   : バイアスに対する勾配 (out_features,) または None
        """
        # dL/dx = grad_output @ W
        grad_input = grad_output @ self.weights

        # dL/dW = grad_output.T @ x  (バッチ全体の和)
        grad_weights = grad_output.t() @ x

        # dL/db = sum_{batch} grad_output
        if self.use_bias:
            grad_bias = grad_output.sum(dim=0)
        else:
            grad_bias = None

        return grad_input, grad_weights, grad_bias


    def update_parameters(self, grad_weights, grad_bias, lr):
        """
        勾配を使ってパラメータを更新（SGD）

        Parameters:
        grad_weights: 重みの勾配 (out_features, in_features)
        grad_bias   : バイアスの勾配 (out_features,) または None
        lr          : 学習率
        """
        self.weights -= lr * grad_weights
        if self.use_bias and grad_bias is not None:
            self.bias -= lr * grad_bias


# --- 以下、テスト用のコード ---

def main():
    in_features = 128
    out_features = 64

    fc = FCLayer(in_features=in_features, out_features=out_features)

    print(f"重みの形状: {fc.weights.shape}")
    print(f"重みの平均: {fc.weights.mean().item():.5f} (理想は 0.0)")
    print(f"重みの標準偏差: {fc.weights.std().item():.5f} (理想は {math.sqrt(2.0 / in_features):.5f})")

    # 順伝播テスト
    x = torch.randn(4, in_features)
    out = fc.forward(x)
    print(f"\n入力形状: {x.shape}")
    print(f"出力形状: {out.shape}")

    # PyTorch の F.linear と順伝播の結果を比較
    import torch.nn.functional as F
    ref = F.linear(x, fc.weights, fc.bias)
    print(f"PyTorch F.linear との最大差: {(out - ref).abs().max().item():.2e} (ほぼ 0 なら正しい)")

    # 逆伝播テスト（PyTorch autograd と比較）
    x_ag = x.clone().requires_grad_(True)
    w_ag = fc.weights.clone().requires_grad_(True)
    b_ag = fc.bias.clone().requires_grad_(True)
    ref_ag = F.linear(x_ag, w_ag, b_ag)
    grad_output = torch.randn_like(ref_ag)
    ref_ag.backward(grad_output)

    grad_input, grad_weights, grad_bias = fc.backward(x, grad_output)

    print(f"\n逆伝播テスト:")
    print(f"  grad_input  最大差: {(grad_input   - x_ag.grad).abs().max().item():.2e}")
    print(f"  grad_weights最大差: {(grad_weights  - w_ag.grad).abs().max().item():.2e}")
    print(f"  grad_bias   最大差: {(grad_bias     - b_ag.grad).abs().max().item():.2e}")


if __name__ == "__main__":
    main()
