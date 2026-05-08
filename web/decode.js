'use strict';

/**
 * decode.js
 * モデル出力のデコードと NMS
 *
 * 出力 boxes 形式: [{ x, y, w, h, score, label }]
 *   x, y: 左上頂点 (正規化 0-1)
 *   w, h: 幅・高さ (正規化 0-1)
 */

const Decoder = (() => {

  // ──────────────────────────────────────────────
  // カスタムCNN デコード (1, 6, 20, 20)
  // ──────────────────────────────────────────────

  /**
   * detection_head.py の decode() に対応するJS実装。
   *
   * チャンネル定義:
   *   ch0: cx_offset (セル内オフセット 0-1)
   *   ch1: cy_offset (セル内オフセット 0-1)
   *   ch2: w  (画像全体に対する正規化幅 0-1)
   *   ch3: h  (画像全体に対する正規化高さ 0-1)
   *   ch4: objectness
   *   ch5: class_prob (face)
   *
   * インデックス: data[ch * GRID*GRID + row * GRID + col]
   */
  function decodeCNN(outputTensor, threshold) {
    const GRID = 20;
    const STRIDE = GRID * GRID;  // 400
    const data = outputTensor.data;

    const candidates = [];
    for (let row = 0; row < GRID; row++) {
      for (let col = 0; col < GRID; col++) {
        const base = row * GRID + col;
        const score = data[4 * STRIDE + base] * data[5 * STRIDE + base];
        if (score < threshold) continue;

        const cx = (col + data[0 * STRIDE + base]) / GRID;
        const cy = (row + data[1 * STRIDE + base]) / GRID;
        const w  = data[2 * STRIDE + base];
        const h  = data[3 * STRIDE + base];

        candidates.push({ x: cx - w / 2, y: cy - h / 2, w, h, score, label: 'face' });
      }
    }
    return nms(candidates, 0.4);
  }


  // ──────────────────────────────────────────────
  // YOLO26n デコード
  // ──────────────────────────────────────────────

  /**
   * YOLO26n ONNX 出力をデコードする。
   *
   * 形状: (1, 300, 6) — ポストNMS済み（NMS はモデル内部に組み込み済み）
   *   各行: [x1, y1, x2, y2, score, class_id]
   *   x1,y1,x2,y2 はピクセル座標 (0-640)
   *   score は信頼度 (0-1)
   *
   * NMS は不要（モデル内部で処理済み）。閾値でフィルタするのみ。
   */
  function decodeYOLO(outputTensor, threshold) {
    const data = outputTensor.data;
    const dims = outputTensor.dims;  // [1, 300, 6]
    if (dims.length !== 3 || dims[2] !== 6) {
      console.warn('YOLO 出力形状が想定外です:', dims);
      return [];
    }

    const numDets = dims[1];
    const boxes   = [];
    for (let i = 0; i < numDets; i++) {
      const base  = i * 6;
      const score = data[base + 4];
      if (score < threshold) continue;

      const x1 = data[base + 0];
      const y1 = data[base + 1];
      const x2 = data[base + 2];
      const y2 = data[base + 3];
      boxes.push({
        x: x1 / 640,
        y: y1 / 640,
        w: (x2 - x1) / 640,
        h: (y2 - y1) / 640,
        score,
        label: 'face',
      });
    }
    return boxes;  // NMS済みなので追加NMS不要
  }


  // ──────────────────────────────────────────────
  // NMS (detection_head.py の nms() に対応)
  // ──────────────────────────────────────────────

  function nms(boxes, iouThreshold) {
    if (!boxes.length) return [];
    boxes.sort((a, b) => b.score - a.score);

    const kept = [];
    const suppressed = new Uint8Array(boxes.length);
    for (let i = 0; i < boxes.length; i++) {
      if (suppressed[i]) continue;
      kept.push(boxes[i]);
      for (let j = i + 1; j < boxes.length; j++) {
        if (!suppressed[j] && iou(boxes[i], boxes[j]) > iouThreshold) {
          suppressed[j] = 1;
        }
      }
    }
    return kept;
  }

  function iou(a, b) {
    const ix = Math.max(0, Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x));
    const iy = Math.max(0, Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y));
    const inter = ix * iy;
    const union = a.w * a.h + b.w * b.h - inter;
    return union < 1e-7 ? 0 : inter / union;
  }


  // ──────────────────────────────────────────────
  // 統合デコード
  // ──────────────────────────────────────────────

  function decode(outputTensor, modelKey, threshold) {
    if (modelKey === 'custom_cnn') return decodeCNN(outputTensor, threshold);
    if (modelKey === 'yolo26n')    return decodeYOLO(outputTensor, threshold);
    return [];
  }

  return { decode };
})();
