import torch
from layers import ConvLayer


class DetectionHead:
    """
    1段階検出器のヘッド（§7.1）。
    1×1 畳み込みで特徴マップの各セルから (cx, cy, w, h, obj, cls...) を予測する。
    """

    def __init__(self, in_channels, num_classes, device=None):
        """
        Parameters:
        in_channels (int): バックボーンの出力チャネル数
        num_classes (int): 検出クラス数（データセットの nc）
        device: パラメータを配置するデバイス（例: torch.device('cuda')）
        """
        self.num_classes = num_classes
        # 各セルの出力: cx, cy, w, h, objectness, class_probs...
        self.num_outputs = 4 + 1 + num_classes
        # 1×1 conv: 各セルを独立して変換（空間情報は保持）
        self.conv = ConvLayer(in_channels, self.num_outputs, kernel_size=1, bias=True, device=device)

    def forward(self, x):
        """
        Parameters:
        x (torch.Tensor): バックボーンの出力 (batch, in_channels, H, W)

        Returns:
        torch.Tensor: (batch, 4+1+num_classes, H, W)（sigmoid 適用済み、全値が [0, 1]）
        """
        raw = self.conv.forward(x)
        return torch.sigmoid(raw)

    def compute_loss(self, pred, targets, lambda1=5.0, lambda2=1.0):
        """
        総合損失 L_total = L_cls + lambda1 * L_box + lambda2 * L_obj を計算する（§7.4）。

        Parameters:
        pred    (torch.Tensor): forward() の出力 (batch, 4+1+C, H, W)
        targets (list[torch.Tensor]): バッチ内の各画像のラベルリスト。
                 各要素は (num_objects, 5) の形で、各行が (class_id, cx, cy, w, h)。
                 cx, cy, w, h は画像全体に対する正規化済み値 [0, 1]。
        lambda1 (float): L_box への重み
        lambda2 (float): L_obj への重み

        Returns:
        tuple: (loss, grad_pred)
            loss     (torch.Tensor): スカラー損失値
            grad_pred(torch.Tensor): pred に対する勾配 (batch, 4+1+C, H, W)
        """
        batch, _, H, W = pred.shape
        eps = 1e-7

        # 目標値テンソルを準備（初期値は全て背景=0）
        target_box = torch.zeros(batch, 4, H, W, device=pred.device)
        target_obj = torch.zeros(batch, 1, H, W, device=pred.device)
        target_cls = torch.zeros(batch, self.num_classes, H, W, device=pred.device)
        obj_mask   = torch.zeros(batch, H, W, dtype=torch.bool, device=pred.device)  # True = 物体あり

        # 各画像のラベルをグリッドに割り当てる
        for b, label in enumerate(targets):
            if label is None or label.numel() == 0:
                continue
            for obj in label:
                cls_id, cx, cy, w, h = obj.tolist()
                cls_id = int(cls_id)
                # 物体中心が含まれるセルを担当セルとする
                gx = min(int(cx * W), W - 1)
                gy = min(int(cy * H), H - 1)
                # cx, cy はセル内オフセット [0, 1)、w, h は画像全体に対する割合
                target_box[b, 0, gy, gx] = cx * W - gx
                target_box[b, 1, gy, gx] = cy * H - gy
                target_box[b, 2, gy, gx] = w
                target_box[b, 3, gy, gx] = h
                target_obj[b, 0, gy, gx] = 1.0
                target_cls[b, cls_id, gy, gx] = 1.0
                obj_mask[b, gy, gx] = True

        pred_box = pred[:, :4]
        pred_obj = pred[:, 4:5]
        pred_cls = pred[:, 5:]

        # --- L_box: 正例セルのみ ---
        # cx, cy は MSE、w, h は sqrt MSE（YOLOv1 方式：小さいボックスの誤差を均等に扱う）
        mask4 = obj_mask.unsqueeze(1).expand_as(pred_box)
        if mask4.any():
            N_pos = obj_mask.sum().item()

            # cx, cy: 標準 MSE
            diff_xy = pred_box[:, :2] - target_box[:, :2]
            L_xy = (diff_xy[mask4[:, :2]] ** 2).sum() / N_pos

            # w, h: sqrt MSE
            eps_sqrt = 1e-6
            sqrt_pred_wh = torch.sqrt(pred_box[:, 2:].clamp(min=eps_sqrt))
            sqrt_tgt_wh  = torch.sqrt(target_box[:, 2:].clamp(min=0))
            diff_wh = sqrt_pred_wh - sqrt_tgt_wh
            L_wh = (diff_wh[mask4[:, 2:]] ** 2).sum() / N_pos

            L_box = L_xy + L_wh

            # 勾配
            grad_xy = torch.zeros_like(pred_box[:, :2])
            grad_xy[mask4[:, :2]] = 2.0 * diff_xy[mask4[:, :2]] / N_pos
            # d/d(p) (sqrt(p) - sqrt(t))^2 = (sqrt(p) - sqrt(t)) / sqrt(p)
            grad_wh = torch.zeros_like(pred_box[:, 2:])
            grad_wh[mask4[:, 2:]] = diff_wh[mask4[:, 2:]] / (sqrt_pred_wh[mask4[:, 2:]] * N_pos)
            grad_box = torch.cat([grad_xy, grad_wh], dim=1)
        else:
            L_box = torch.tensor(0.0, device=pred.device)
            grad_box = torch.zeros_like(pred_box)

        # --- L_obj: 全セル BCE（背景セルは noobj_scale で重みを下げる）---
        # YOLOv1 に倣い 0.5 とする。0.0316（損失総量均等化）では背景セルへの
        # 勾配が弱すぎて Precision が崩壊することが判明したため変更。
        noobj_scale = 0.5
        L_obj = -(target_obj * torch.log(pred_obj + eps)
                  + noobj_scale * (1 - target_obj) * torch.log(1 - pred_obj + eps)).mean()
        N_obj = pred_obj.numel()
        grad_obj = (-(target_obj / (pred_obj + eps))
                    + noobj_scale * (1 - target_obj) / (1 - pred_obj + eps)) / N_obj

        # --- L_cls: 正例セルのみ BCE ---
        mask_cls = obj_mask.unsqueeze(1).expand_as(pred_cls)
        if mask_cls.any():
            p_pos = pred_cls[mask_cls]
            t_pos = target_cls[mask_cls]
            L_cls = -(t_pos * torch.log(p_pos + eps)
                      + (1 - t_pos) * torch.log(1 - p_pos + eps)).mean()
            N_cls = p_pos.numel()
            grad_cls = torch.zeros_like(pred_cls)
            grad_cls[mask_cls] = (-(t_pos / (p_pos + eps))
                                  + (1 - t_pos) / (1 - p_pos + eps)) / N_cls
        else:
            L_cls = torch.tensor(0.0, device=pred.device)
            grad_cls = torch.zeros_like(pred_cls)

        L_total = L_cls + lambda1 * L_box + lambda2 * L_obj

        # 勾配を結合して pred と同じ形状にする
        grad_pred = torch.cat([
            lambda1 * grad_box,
            lambda2 * grad_obj,
            grad_cls,
        ], dim=1)

        return L_total, grad_pred

    def backward(self, x, pred, grad_pred):
        """
        sigmoid と conv を通じて勾配を逆伝播する。

        Parameters:
        x        (torch.Tensor): forward() に渡した入力 (batch, in_channels, H, W)
        pred     (torch.Tensor): forward() の出力（sigmoid 適用済み）
        grad_pred(torch.Tensor): compute_loss() が返した pred への勾配

        Returns:
        tuple: (grad_input, grad_weights, grad_bias)
        """
        # sigmoid の逆伝播: d(sigmoid)/dz = sigmoid(z) * (1 - sigmoid(z)) = pred * (1 - pred)
        grad_raw = grad_pred * pred * (1 - pred)
        # conv の逆伝播
        return self.conv.backward(x, grad_raw)

    def update_parameters(self, grad_weights, grad_bias, lr):
        self.conv.update_parameters(grad_weights, grad_bias, lr)

    def decode(self, pred):
        """
        セル相対座標を画像全体の正規化座標に変換する（推論時に使用）。

        Parameters:
        pred (torch.Tensor): forward() の出力 (batch, 4+1+C, H, W)

        Returns:
        tuple: (cx, cy, w, h, obj, cls)  各テンソルは (batch, H, W) または (batch, C, H, W)
        """
        _, _, H, W = pred.shape
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, dtype=torch.float32, device=pred.device),
            torch.arange(W, dtype=torch.float32, device=pred.device),
            indexing='ij',
        )
        cx  = (pred[:, 0] + grid_x) / W
        cy  = (pred[:, 1] + grid_y) / H
        w   = pred[:, 2]
        h   = pred[:, 3]
        obj = pred[:, 4]
        cls = pred[:, 5:]
        return cx, cy, w, h, obj, cls


def nms(boxes, scores, iou_threshold=0.4, score_threshold=0.5):
    """
    Non-Maximum Suppression（§7.3）。

    Parameters:
    boxes         (torch.Tensor): (N, 4) の形で (cx, cy, w, h) が正規化済み
    scores        (torch.Tensor): (N,) の信頼度スコア
    iou_threshold (float): IoU のしきい値（これを超える枠を削除）
    score_threshold(float): スコアのしきい値（これ未満の枠を最初に削除）

    Returns:
    list[int]: 残った枠のインデックスリスト
    """
    # ステップ1: スコアによる足切り
    keep_mask = scores >= score_threshold
    indices = keep_mask.nonzero(as_tuple=False).squeeze(1)
    if indices.numel() == 0:
        return []

    boxes  = boxes[indices]
    scores = scores[indices]

    # cx, cy, w, h → x1, y1, x2, y2 に変換（IoU 計算用）
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    areas = (x2 - x1) * (y2 - y1)

    # ステップ2: スコア順にソート（降順）
    order = scores.argsort(descending=True)

    kept = []
    while order.numel() > 0:
        # ステップ3: 最大スコアの枠を確定
        i = order[0].item()
        kept.append(indices[i].item())

        if order.numel() == 1:
            break

        # ステップ4: 確定枠と残りの枠の IoU を計算
        rest = order[1:]
        inter_x1 = torch.clamp(x1[rest], min=x1[i])
        inter_y1 = torch.clamp(y1[rest], min=y1[i])
        inter_x2 = torch.clamp(x2[rest], max=x2[i])
        inter_y2 = torch.clamp(y2[rest], max=y2[i])
        inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
        inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
        intersection = inter_w * inter_h
        union = areas[i] + areas[rest] - intersection
        iou = intersection / (union + 1e-7)

        # IoU がしきい値以下の枠だけ残す
        order = rest[iou <= iou_threshold]

    return kept


# --- 以下、テスト用のコード ---

def main():
    import math

    in_channels = 32
    num_classes = 1  # 顔検出データセット（nc=1）
    H, W = 20, 20   # 640px / stride 32 = 20

    head = DetectionHead(in_channels=in_channels, num_classes=num_classes)
    print(f"出力チャネル数: {head.num_outputs}  (4 + 1 + {num_classes})")

    # 順伝播テスト
    batch = 2
    x = torch.randn(batch, in_channels, H, W)
    pred = head.forward(x)
    print(f"\n入力形状: {x.shape}")
    print(f"出力形状: {pred.shape}")
    print(f"出力の最小値: {pred.min().item():.4f}  最大値: {pred.max().item():.4f}  (全て [0,1] なら正しい)")

    # ダミーラベル: 画像0に物体1個（クラス0、中心0.5,0.5、幅高さ0.2x0.3）
    targets = [
        torch.tensor([[0.0, 0.5, 0.5, 0.2, 0.3]]),
        torch.zeros(0, 5),  # 画像1は物体なし
    ]

    loss, grad_pred = head.compute_loss(pred, targets)
    print(f"\n総合損失: {loss.item():.4f}")
    print(f"grad_pred の形状: {grad_pred.shape}")

    # 損失の数値微分との比較（compute_loss の勾配検証）
    delta = 1e-4
    idx = (0, 0, 10, 10)  # バッチ0, チャネル0(cx), セル(10,10)
    pred_plus  = pred.clone(); pred_plus[idx]  += delta
    pred_minus = pred.clone(); pred_minus[idx] -= delta
    loss_plus,  _ = head.compute_loss(pred_plus,  targets)
    loss_minus, _ = head.compute_loss(pred_minus, targets)
    numerical_grad = (loss_plus - loss_minus).item() / (2 * delta)
    analytical_grad = grad_pred[idx].item()
    print(f"\n勾配検証 (セル[0,0,10,10]):")
    print(f"  数値微分: {numerical_grad:.6f}")
    print(f"  解析的勾配: {analytical_grad:.6f}")
    print(f"  相対誤差: {abs(numerical_grad - analytical_grad) / (abs(numerical_grad) + 1e-8):.2e}")

    # decode のテスト
    cx, cy, w, h, obj, cls = head.decode(pred)
    print(f"\ndecode 後:")
    print(f"  cx の範囲: [{cx.min().item():.3f}, {cx.max().item():.3f}]")
    print(f"  cy の範囲: [{cy.min().item():.3f}, {cy.max().item():.3f}]")

    # NMS のテスト
    test_boxes  = torch.tensor([[0.5, 0.5, 0.2, 0.2],
                                 [0.51, 0.51, 0.2, 0.2],  # ほぼ同じ位置 → 抑制される
                                 [0.9, 0.9, 0.1, 0.1]])   # 離れた位置 → 残る
    test_scores = torch.tensor([0.9, 0.8, 0.7])
    kept = nms(test_boxes, test_scores, iou_threshold=0.4, score_threshold=0.5)
    print(f"\nNMS テスト: 入力 3 枠 → 残存インデックス {kept}  (期待値: [0, 2])")


if __name__ == "__main__":
    main()
