'use strict';

/**
 * inference.js
 * ONNX Runtime Web を使ったモデルセッション管理と推論
 *
 * WebGPU EP → WASM EP (シングルスレッド) の順でフォールバックする。
 * SharedArrayBuffer を使わないためCOOP/COEPヘッダーは不要。
 */

// COOP/COEP ヘッダーで SharedArrayBuffer が有効なら ORT は内部 Worker を使用し
// WASM 推論がメインスレッドをブロックしなくなる。
// numThreads は ORT に自動判定させる（SharedArrayBuffer 有無で切り替わる）。
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';

const InferenceEngine = (() => {
  let _session     = null;
  let _currentKey  = null;
  let _backend     = null;
  let _manifest    = null;
  let _loading     = false;

  async function _loadManifest() {
    if (_manifest) return _manifest;
    const res = await fetch('./models/manifest.json');
    if (!res.ok) throw new Error('manifest.json の読み込みに失敗しました');
    _manifest = await res.json();
    return _manifest;
  }

  async function _createSession(url) {
    const gpuAvailable = typeof navigator !== 'undefined' && 'gpu' in navigator;

    if (gpuAvailable) {
      try {
        const session = await ort.InferenceSession.create(url, {
          executionProviders: ['webgpu'],
          graphOptimizationLevel: 'all',
        });
        return { session, backend: 'webgpu' };
      } catch (e) {
        console.warn('WebGPU EP 失敗、WASM にフォールバック:', e.message);
      }
    }

    const session = await ort.InferenceSession.create(url, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    });
    return { session, backend: 'wasm' };
  }

  /**
   * モデルを切り替える。既存セッションを解放してから新規作成する。
   * @param {string} modelKey - 'custom_cnn' | 'yolo26n'
   * @returns {string} backend ('webgpu' | 'wasm')
   */
  async function switchModel(modelKey) {
    _loading = true;

    if (_session) {
      try { await _session.release(); } catch (_) { /* ignore */ }
      _session = null;
    }

    const manifest = await _loadManifest();
    const info = manifest[modelKey];
    if (!info) throw new Error(`不明なモデル: ${modelKey}`);

    const { session, backend } = await _createSession(`./models/${info.file}`);
    _session    = session;
    _currentKey = modelKey;
    _backend    = backend;
    _loading    = false;
    return backend;
  }

  /**
   * 推論を実行する。
   * offscreenCanvas は 640×640 で main.js 側が描画済みであること。
   *
   * 前処理: RGBA ImageData → NCHW Float32Array [0,1]
   *   (training: transforms.ToTensor() のみ、letterboxなし、stretchリサイズ)
   *
   * @param {HTMLCanvasElement} offscreen640 - 640×640 の作業用 Canvas
   * @returns {{ outputTensor: ort.Tensor, modelKey: string } | null}
   */
  async function runInference(offscreen640) {
    if (!_session || _loading) return null;

    const ctx = offscreen640.getContext('2d');
    const { data } = ctx.getImageData(0, 0, 640, 640);
    const N = 640 * 640;
    const float32 = new Float32Array(3 * N);

    for (let i = 0; i < N; i++) {
      float32[0 * N + i] = data[i * 4]     / 255;
      float32[1 * N + i] = data[i * 4 + 1] / 255;
      float32[2 * N + i] = data[i * 4 + 2] / 255;
    }

    const tensor = new ort.Tensor('float32', float32, [1, 3, 640, 640]);
    const inputName = _session.inputNames[0];
    const results = await _session.run({ [inputName]: tensor });
    const outputTensor = results[_session.outputNames[0]];

    return { outputTensor, modelKey: _currentKey };
  }

  function getBackend()    { return _backend; }
  function isLoading()     { return _loading; }
  function getCurrentKey() { return _currentKey; }

  return { switchModel, runInference, getBackend, isLoading, getCurrentKey };
})();
