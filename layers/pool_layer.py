import torch
import torch.nn.functional as F


class PoolLayer:
    def __init__(self, kernel_size, stride=None, mode='max'):
        """
        プーリング層の初期化

        Parameters:
        kernel_size (int): プーリングウィンドウのサイズ
        stride (int): ストライド（デフォルトは kernel_size と同じ）
        mode (str): プーリングの種類。'max'（最大値）または 'avg'（平均値）
        """
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.mode = mode


    def forward(self, x):
        """
        順伝播

        Parameters:
        x (torch.Tensor): 入力テンソル (batch_size, channels, height, width)

        Returns:
        torch.Tensor: 出力テンソル (batch_size, channels, out_height, out_width)
        """
        batch, channels, height, width = x.shape
        out_h = (height - self.kernel_size) // self.stride + 1
        out_w = (width  - self.kernel_size) // self.stride + 1
        k2 = self.kernel_size * self.kernel_size

        # ------------------------------------
        # --- im2col変換でウィンドウを行列に展開 ---
        # ------------------------------------

        # F.unfold は im2col 処理を行う関数。各ウィンドウの要素を列に並べる
        # (batch, channels*k*k, out_h*out_w)
        x_col = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride)

        # チャネルごとに独立して扱えるよう分離
        # (batch, channels, k*k, out_h*out_w)
        x_windows = x_col.view(batch, channels, k2, out_h * out_w)

        if self.mode == 'max':
            out = x_windows.max(dim=2).values
        elif self.mode == 'avg':
            out = x_windows.mean(dim=2)
        else:
            raise ValueError(f"未対応のプーリング種別: {self.mode}")

        return out.view(batch, channels, out_h, out_w)


    def backward(self, x, grad_output):
        """
        逆伝播

        Parameters:
        x (torch.Tensor): 順伝播時の入力テンソル (batch_size, channels, height, width)
        grad_output (torch.Tensor): 出力側からの勾配 (batch_size, channels, out_height, out_width)

        Returns:
        torch.Tensor: 入力に対する勾配 (batch_size, channels, height, width)
        """
        batch, channels, height, width = x.shape
        out_h, out_w = grad_output.shape[2], grad_output.shape[3]
        k2 = self.kernel_size * self.kernel_size

        # ------------------------------------
        # --- im2col変換でウィンドウを行列に展開 ---
        # ------------------------------------

        # F.unfold で im2col 変換。F.fold（col2im）と要素の並び順が一致することが保証される
        # (batch, channels*k*k, out_h*out_w)
        x_col = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride)

        # batch と channels を統合して処理しやすくする
        # (batch*channels, k*k, out_h*out_w)
        x_windows = x_col.view(batch, channels, k2, out_h * out_w).reshape(batch * channels, k2, out_h * out_w)

        # 勾配も同様に統合: (batch*channels, 1, out_h*out_w)
        grad_flat = grad_output.reshape(batch * channels, 1, out_h * out_w)

        if self.mode == 'max':
            # 最大値を取った位置にだけ勾配を流す
            max_idx = x_windows.argmax(dim=1, keepdim=True)
            mask = torch.zeros_like(x_windows).scatter_(1, max_idx, 1.0)
            grad_windows = mask * grad_flat

        elif self.mode == 'avg':
            # ウィンドウ内の全位置に均等に勾配を分配する
            grad_windows = grad_flat.expand(-1, k2, -1) / k2

        else:
            raise ValueError(f"未対応のプーリング種別: {self.mode}")

        # F.fold で重複する位置の勾配を足し合わせながら元のサイズに戻す
        # reshape で (batch, channels*k*k, out_h*out_w) に戻してから fold
        grad_col = grad_windows.view(batch, channels * k2, out_h * out_w)
        return F.fold(grad_col,
                      output_size=(height, width),
                      kernel_size=self.kernel_size,
                      stride=self.stride)


# --- 以下、テスト用のコード ---

def test(mode, kernel_size, stride):
    pool = PoolLayer(kernel_size=kernel_size, stride=stride, mode=mode)
    x = torch.randn(2, 3, 8, 8)

    # 順伝播テスト
    out = pool.forward(x)
    if mode == 'max':
        ref = F.max_pool2d(x, kernel_size=kernel_size, stride=stride)
    else:
        ref = F.avg_pool2d(x, kernel_size=kernel_size, stride=stride)
    fwd_diff = (out - ref).abs().max().item()

    # 逆伝播テスト（PyTorch autograd と比較）
    x_ag = x.clone().requires_grad_(True)
    if mode == 'max':
        ref_ag = F.max_pool2d(x_ag, kernel_size=kernel_size, stride=stride)
    else:
        ref_ag = F.avg_pool2d(x_ag, kernel_size=kernel_size, stride=stride)
    grad_output = torch.randn_like(ref_ag)
    ref_ag.backward(grad_output)

    grad_input = pool.backward(x, grad_output)
    bwd_diff = (grad_input - x_ag.grad).abs().max().item()

    print(f"[{mode:3s}] kernel={kernel_size}, stride={stride}  "
          f"順伝播最大差: {fwd_diff:.2e}  逆伝播最大差: {bwd_diff:.2e}")


def main():
    # 非重複（標準的なプーリング）
    test('max', kernel_size=2, stride=2)
    test('avg', kernel_size=2, stride=2)

    # 重複あり（勾配の蓄積が発生するケース）
    test('max', kernel_size=3, stride=2)
    test('avg', kernel_size=3, stride=2)


if __name__ == "__main__":
    main()
