import torch
import torch.nn.functional as F


class ActivationLayer:
    def __init__(self, activation='relu'):
        """
        活性化層の初期化

        Parameters:
        activation (str): 活性化関数の種類（現在は 'relu' のみ対応）
        """
        self.activation = activation


    def forward(self, x):
        """
        順伝播

        Parameters:
        x (torch.Tensor): 入力テンソル（任意の形状）

        Returns:
        torch.Tensor: 活性化関数を適用した出力テンソル（入力と同じ形状）
        """
        if self.activation == 'relu':
            return torch.clamp(x, min=0)

        raise ValueError(f"未対応の活性化関数: {self.activation}")


    def backward(self, x, grad_output):
        """
        逆伝播

        Parameters:
        x (torch.Tensor): 順伝播時の入力テンソル
        grad_output (torch.Tensor): 出力側からの勾配（入力と同じ形状）

        Returns:
        torch.Tensor: 入力に対する勾配（入力と同じ形状）
        """
        if self.activation == 'relu':
            # 順伝播時に正だった位置にだけ勾配を流す
            return grad_output * (x > 0)

        raise ValueError(f"未対応の活性化関数: {self.activation}")


# --- 以下、テスト用のコード ---

def main():
    activation = ActivationLayer(activation='relu')

    # 負の値を含む入力で ReLU の動作を確認
    x = torch.randn(2, 4, 6, 6)

    # 順伝播テスト
    out = activation.forward(x)
    ref = F.relu(x)
    print(f"順伝播 最大差: {(out - ref).abs().max().item():.2e} (ほぼ 0 なら正しい)")

    # 逆伝播テスト（PyTorch autograd と比較）
    x_ag = x.clone().requires_grad_(True)
    ref_ag = F.relu(x_ag)
    grad_output = torch.randn_like(ref_ag)
    ref_ag.backward(grad_output)

    grad_input = activation.backward(x, grad_output)
    print(f"逆伝播 最大差: {(grad_input - x_ag.grad).abs().max().item():.2e} (ほぼ 0 なら正しい)")


if __name__ == "__main__":
    main()
