'use strict';

/**
 * main.js
 * カメラ制御、推論ループ、Canvas描画
 */

(async () => {
  const video          = document.getElementById('video');
  const canvas         = document.getElementById('canvas');
  const ctx            = canvas.getContext('2d');
  const fpsEl          = document.getElementById('fps');
  const backendBadge   = document.getElementById('backend-badge');
  const loadingOverlay = document.getElementById('loading-overlay');
  const loadingMsg     = document.getElementById('loading-msg');
  const thresholdInput = document.getElementById('threshold');
  const thresholdVal   = document.getElementById('threshold-val');
  const btnCNN         = document.getElementById('btn-cnn');
  const btnYOLO        = document.getElementById('btn-yolo');

  let threshold     = 0.5;
  let rafId         = null;
  let lastTime      = performance.now();
  let fpsAccum      = 0;
  let fpsCount      = 0;

  // 640×640 オフスクリーン Canvas を1個作成して毎フレーム再利用（GC抑制）
  const offscreen = document.createElement('canvas');
  offscreen.width = offscreen.height = 640;
  const offCtx = offscreen.getContext('2d');

  // ──────────────── カメラ初期化 ────────────────
  async function initCamera() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
        audio: false,
      });
      video.srcObject = stream;
      await new Promise(r => { video.onloadedmetadata = r; });
      canvas.width  = video.videoWidth  || 640;
      canvas.height = video.videoHeight || 480;
      // canvas-container の高さをビデオに合わせる
      document.getElementById('canvas-container').style.height = canvas.height + 'px';
    } catch {
      alert('カメラへのアクセスが拒否されました。ブラウザの設定をご確認ください。');
      throw new Error('camera denied');
    }
  }

  // ──────────────── モデル切り替え ────────────────
  window.switchModel = async function (modelKey) {
    if (InferenceEngine.isLoading()) return;

    btnCNN.disabled  = true;
    btnYOLO.disabled = true;
    loadingMsg.textContent = modelKey === 'custom_cnn'
      ? 'Custom CNN を読み込み中...'
      : 'YOLO26n を読み込み中...';
    loadingOverlay.classList.remove('hidden');

    if (rafId !== null) { cancelAnimationFrame(rafId); rafId = null; }

    try {
      const backend = await InferenceEngine.switchModel(modelKey);

      backendBadge.textContent = backend.toUpperCase();
      backendBadge.className   = 'badge ' + backend;
      btnCNN.classList.toggle('active',  modelKey === 'custom_cnn');
      btnYOLO.classList.toggle('active', modelKey === 'yolo26n');
    } catch (e) {
      console.error(e);
      alert('モデルの読み込みに失敗しました。コンソールをご確認ください。');
    } finally {
      btnCNN.disabled  = false;
      btnYOLO.disabled = false;
      loadingOverlay.classList.add('hidden');
      lastTime = performance.now();
      rafId = requestAnimationFrame(loop);
    }
  };

  // ──────────────── 推論ループ ────────────────
  async function loop() {
    // 640×640 にstretchしてオフスクリーンCanvasへ描画
    offCtx.drawImage(video, 0, 0, 640, 640);

    let boxes = [];
    try {
      const result = await InferenceEngine.runInference(offscreen);
      if (result) {
        boxes = Decoder.decode(result.outputTensor, result.modelKey, threshold);
      }
    } catch (e) {
      console.warn('推論エラー:', e);
    }

    // 表示 Canvas にビデオフレームを描画（表示サイズに合わせてstretch）
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    drawBoxes(boxes);

    // FPS（1秒平均）
    const now = performance.now();
    fpsAccum += now - lastTime;
    lastTime  = now;
    fpsCount++;
    if (fpsAccum >= 1000) {
      fpsEl.textContent = fpsCount.toFixed(0);
      fpsAccum = 0;
      fpsCount = 0;
    }

    rafId = requestAnimationFrame(loop);
  }

  // ──────────────── Canvas 描画 ────────────────
  function drawBoxes(boxes) {
    const W = canvas.width;
    const H = canvas.height;

    for (const box of boxes) {
      const px = box.x * W;
      const py = box.y * H;
      const pw = box.w * W;
      const ph = box.h * H;
      const label = `face ${(box.score * 100).toFixed(0)}%`;

      ctx.strokeStyle = '#00e676';
      ctx.lineWidth   = 2;
      ctx.strokeRect(px, py, pw, ph);

      ctx.font = 'bold 12px monospace';
      const tw = ctx.measureText(label).width;
      ctx.fillStyle = 'rgba(0,0,0,0.65)';
      ctx.fillRect(px, Math.max(0, py - 18), tw + 6, 18);
      ctx.fillStyle = '#00e676';
      ctx.fillText(label, px + 3, Math.max(13, py - 4));
    }
  }

  // ──────────────── スライダー ────────────────
  thresholdInput.addEventListener('input', () => {
    threshold = parseFloat(thresholdInput.value);
    thresholdVal.textContent = threshold.toFixed(2);
  });

  // ──────────────── 起動 ────────────────
  await initCamera();
  await switchModel('custom_cnn');
})();
