import torch
import torch.nn.functional as F


class DropoutLayer:
    def __init__(self, p=0.5):
        """
        ドロップアウト層の初期化

        Parameters:
        p (float): ニューロンを無効化する確率（デフォルトは 0.5）
        """
        if not 0.0 <= p < 1.0:
            raise ValueError(f"p は 0 以上 1 未満の値でなければなりません: {p}")
        self.p = p
        self.training = True
        self.mask = None


    def forward(self, x):
        """
        順伝播

        Parameters:
        x (torch.Tensor): 入力テンソル（任意の形状）

        Returns:
        torch.Tensor: ドロップアウトを適用した出力テンソル（入力と同じ形状）
        """
        if not self.training:
            return x

        # 各要素を確率 (1-p) で残すベルヌーイマスクを生成
        self.mask = (torch.rand_like(x) >= self.p).float()

        # Inverted dropout: スケールして推論時に補正不要にする
        return x * self.mask / (1.0 - self.p)


    def backward(self, grad_output):
        """
        逆伝播

        Parameters:
        grad_output (torch.Tensor): 出力側からの勾配（入力と同じ形状）

        Returns:
        torch.Tensor: 入力に対する勾配（入力と同じ形状）
        """
        if not self.training:
            return grad_output

        # 順伝播で使ったマスクをそのまま適用してスケール
        return grad_output * self.mask / (1.0 - self.p)


# --- 以下、テスト用のコード ---

def main():
    p = 0.5
    dropout = DropoutLayer(p=p)

    x = torch.randn(4, 128)

    # --- 訓練モードのテスト ---
    dropout.training = True
    torch.manual_seed(0)
    out = dropout.forward(x)
    zero_ratio = (out == 0).float().mean().item()
    print(f"訓練モード:")
    print(f"  入力形状: {x.shape}")
    print(f"  出力形状: {out.shape}")
    print(f"  ゼロ要素の割合: {zero_ratio:.3f} (期待値 ≈ {p:.1f})")
    print(f"  非ゼロ要素のスケール確認（元値との比）:")
    nonzero_mask = out != 0
    ratio = (out[nonzero_mask] / x[nonzero_mask]).mean().item()
    print(f"    平均比率 = {ratio:.4f} (期待値 = {1.0 / (1.0 - p):.4f})")

    # 逆伝播テスト（PyTorch autograd と比較）
    x_ag = x.clone().requires_grad_(True)
    # PyTorch の dropout は同じシードでも内部乱数が異なるため、
    # 自作マスクを autograd グラフに埋め込んで比較する
    mask_ref = dropout.mask.clone()
    ref_out = x_ag * mask_ref / (1.0 - p)
    grad_output = torch.randn_like(ref_out)
    ref_out.backward(grad_output)

    grad_input = dropout.backward(grad_output)
    print(f"\n  逆伝播 最大差: {(grad_input - x_ag.grad).abs().max().item():.2e} (ほぼ 0 なら正しい)")

    # --- 推論モードのテスト ---
    dropout.training = False
    out_eval = dropout.forward(x)
    print(f"\n推論モード:")
    print(f"  入出力の最大差: {(out_eval - x).abs().max().item():.2e} (ほぼ 0 なら正しい、恒等変換のため)")

    grad_eval = dropout.backward(grad_output)
    print(f"  逆伝播 最大差: {(grad_eval - grad_output).abs().max().item():.2e} (ほぼ 0 なら正しい、恒等変換のため)")


if __name__ == "__main__":
    main()
