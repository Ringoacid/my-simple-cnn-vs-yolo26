import torch
import torch.nn.functional as F
import math

class ConvLayer():
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, device=None):
        """
        畳み込み層の初期化

        Parameters:
        in_channels (int): 入力チャネル数
        out_channels (int): 出力チャネル数
        kernel_size (int): カーネルサイズ（例: 3なら3x3のカーネル）
        stride (int): ストライド（デフォルトは1）
        padding (int): パディング（デフォルトは0）
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.device = device

        # カーネルの重みをHeの初期化で生成
        self.weights = self.he_initialization()

        # バイアスの初期化
        self.use_bias = bias
        if self.use_bias:
            self.bias = self.bias_initialization()
        else:
            self.bias = None


    def he_initialization(self):
        """
        Heの初期化（Kaiming Normal）による重みテンソルの生成
        """
        # 1. 1つのカーネルの入力ノード数を計算
        fan_in = self.in_channels * self.kernel_size * self.kernel_size
        
        # 2. 標準偏差を計算
        sigma = math.sqrt(2.0 / fan_in)
        
        # 3. 重みテンソルの形状を定義
        #  (out_channels, in_channels, kernel_height, kernel_width) 
        weight_shape = (self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        
        # 4. 平均0、分散1の標準正規分布から生成したテンソルに sigma を掛けてスケーリングする
        weights = torch.randn(weight_shape, device=self.device) * sigma

        return weights


    def bias_initialization(self):
        """
        バイアスの初期化を行う関数
        """
        fan_in = self.in_channels * self.kernel_size * self.kernel_size
        bound = 1 / math.sqrt(fan_in)
        
        # 出力チャンネル数分のバイアスを Uniform(-bound, bound) で生成
        # torch.rand は [0, 1) なので、スケーリングして [-bound, bound) に変換
        bias = (torch.rand(self.out_channels, device=self.device) * 2 - 1) * bound
        return bias


    def forward(self, x):
        """
        順伝播

        Parameters:
        x (torch.Tensor): 前の層からの入力テンソル (batch_size, in_channels, height, width)

        Returns:
        torch.Tensor: 出力テンソル (batch_size, out_channels, out_height, out_width)
        """
        batch_size, in_channels, height, width = x.shape

        # ゼロパディング
        if self.padding > 0:
            # まず、パディング後のサイズの0テンソルを作成
            x_padded = torch.zeros(batch_size, in_channels,
                                   height + 2 * self.padding,
                                   width + 2 * self.padding, device=x.device)

            # 必要な場所に、元の入力テンソルをコピー
            x_padded[:, :, self.padding:self.padding + height,
                           self.padding:self.padding + width] = x
        else:
            x_padded = x

        # 出力サイズを計算
        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width  = (width  + 2 * self.padding - self.kernel_size) // self.stride + 1

        # ------------------------------------
        # --- im2col変換で、畳み込みを行列積に変換 ---
        # ------------------------------------

        # unfold 後: (batch, in_channels, out_height, out_width, kernel_size, kernel_size)
        x_unfolded = (x_padded
                      .unfold(2, self.kernel_size, self.stride)
                      .unfold(3, self.kernel_size, self.stride))
        # カーネル次元を空間次元の前に移動して正しい順序に並べる
        # (batch, in_channels, kernel_size, kernel_size, out_height, out_width)
        #  → (batch, in_channels * kernel_size * kernel_size, out_height * out_width)
        x_col = (x_unfolded
                 .permute(0, 1, 4, 5, 2, 3)
                 .contiguous()
                 .view(batch_size, in_channels * self.kernel_size * self.kernel_size, -1))

        # 重みを行列形式に変換: (out_channels, in_channels * kernel_size * kernel_size)
        w = self.weights.view(self.out_channels, -1)

        # 行列積: (batch, out_channels, out_height * out_width)
        out = torch.matmul(w, x_col)

        if self.use_bias:
            out = out + self.bias.view(1, self.out_channels, 1)

        return out.view(batch_size, self.out_channels, out_height, out_width)


    def update_parameters(self, grad_weights, grad_bias, lr):
        """
        勾配を使ってパラメータを更新（SGD）

        Parameters:
        grad_weights: 重みの勾配 (out_channels, in_channels, kernel_size, kernel_size)
        grad_bias   : バイアスの勾配 (out_channels,) または None
        lr          : 学習率
        """
        self.weights -= lr * grad_weights
        if self.use_bias and grad_bias is not None:
            self.bias -= lr * grad_bias


    def backward(self, x, grad_output):
        """
        逆伝播

        Parameters:
        x (torch.Tensor): 順伝播時の入力テンソル (batch_size, in_channels, height, width)
        grad_output (torch.Tensor): 出力側からの勾配 (batch_size, out_channels, out_height, out_width)

        Returns:
        tuple: (grad_input, grad_weights, grad_bias)
            grad_input  : 入力に対する勾配 (batch_size, in_channels, height, width)
            grad_weights: 重みに対する勾配 (out_channels, in_channels, kernel_size, kernel_size)
            grad_bias   : バイアスに対する勾配 (out_channels,) または None
        """
        batch_size, in_channels, height, width = x.shape
        _, out_channels, out_height, out_width = grad_output.shape

        # ゼロパディング（forward と同じ処理）
        if self.padding > 0:
            x_padded = torch.zeros(batch_size, in_channels,
                                   height + 2 * self.padding,
                                   width + 2 * self.padding, device=x.device)
            x_padded[:, :, self.padding:self.padding + height,
                           self.padding:self.padding + width] = x
        else:
            x_padded = x

        # im2col 変換（forward と同じ処理）
        x_unfolded = (x_padded
                      .unfold(2, self.kernel_size, self.stride)
                      .unfold(3, self.kernel_size, self.stride))
        x_col = (x_unfolded
                 .permute(0, 1, 4, 5, 2, 3)
                 .contiguous()
                 .view(batch_size, in_channels * self.kernel_size * self.kernel_size, -1))

        # grad_output を (batch, out_channels, out_height * out_width) に整形
        grad_out_col = grad_output.view(batch_size, out_channels, -1)

        # --- バイアスの勾配 ---
        # dL/db[oc] = sum_{batch, oh, ow} grad_output[b, oc, oh, ow]
        if self.use_bias:
            grad_bias = grad_out_col.sum(dim=(0, 2))
        else:
            grad_bias = None

        # --- 重みの勾配 ---
        # dL/dW = sum_b grad_out_col[b] @ x_col[b].T
        # (batch, out_c, out_h*out_w) @ (batch, out_h*out_w, in_c*k*k) → sum → (out_c, in_c*k*k)
        grad_weights = torch.matmul(grad_out_col, x_col.permute(0, 2, 1)).sum(dim=0)
        grad_weights = grad_weights.view(self.out_channels, self.in_channels,
                                         self.kernel_size, self.kernel_size)

        # --- 入力の勾配 ---
        # dx_col = W.T @ grad_out_col
        # (in_c*k*k, out_c) @ (batch, out_c, out_h*out_w) → (batch, in_c*k*k, out_h*out_w)
        w_col = self.weights.view(self.out_channels, -1)
        dx_col = torch.matmul(w_col.t(), grad_out_col)

        # col2im: fold で (batch, in_c*k*k, out_h*out_w) → (batch, in_c, h_padded, w_padded)
        h_padded = height + 2 * self.padding
        w_padded = width + 2 * self.padding
        dx_padded = F.fold(dx_col,
                           output_size=(h_padded, w_padded),
                           kernel_size=self.kernel_size,
                           stride=self.stride)

        # パディング分を除去して元の入力サイズに戻す
        if self.padding > 0:
            grad_input = dx_padded[:, :, self.padding:self.padding + height,
                                         self.padding:self.padding + width]
        else:
            grad_input = dx_padded

        return grad_input, grad_weights, grad_bias


# --- 以下、テスト用のコード ---

def main():
    in_channels = 3
    out_channels = 64
    kernel_size = 3

    conv = ConvLayer(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)

    print(f"重みの形状: {conv.weights.shape}")
    print(f"重みの平均: {conv.weights.mean().item():.5f} (理想は 0.0)")
    print(f"重みの標準偏差: {conv.weights.std().item():.5f} (理想は {math.sqrt(2.0 / (in_channels * kernel_size * kernel_size)):.5f})")

    # 順伝播テスト
    x = torch.randn(1, in_channels, 8, 8)
    out = conv.forward(x)
    print(f"\n入力形状: {x.shape}")
    print(f"出力形状: {out.shape}")

    # PyTorch の F.conv2d と順伝播の結果を比較
    ref = F.conv2d(x, conv.weights, conv.bias, stride=conv.stride, padding=conv.padding)
    max_diff = (out - ref).abs().max().item()
    print(f"PyTorch F.conv2d との最大差: {max_diff:.2e} (ほぼ 0 なら正しい)")

    # 逆伝播テスト（PyTorch autograd と比較）
    x_ag = x.clone().requires_grad_(True)
    w_ag = conv.weights.clone().requires_grad_(True)
    b_ag = conv.bias.clone().requires_grad_(True)
    ref_ag = F.conv2d(x_ag, w_ag, b_ag, stride=conv.stride, padding=conv.padding)
    grad_output = torch.randn_like(ref_ag)
    ref_ag.backward(grad_output)

    grad_input, grad_weights, grad_bias = conv.backward(x, grad_output)

    print(f"\n逆伝播テスト:")
    print(f"  grad_input  最大差: {(grad_input  - x_ag.grad ).abs().max().item():.2e}")
    print(f"  grad_weights最大差: {(grad_weights - w_ag.grad ).abs().max().item():.2e}")
    print(f"  grad_bias   最大差: {(grad_bias   - b_ag.grad ).abs().max().item():.2e}")


if __name__ == "__main__":
    main()