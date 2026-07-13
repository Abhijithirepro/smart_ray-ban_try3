/* Glasses-region CNN inference via onnxruntime-web. Given the located region crop
 * (an RGBA cv.Mat), resize to the model input, build a normalized NCHW RGB tensor,
 * run the ONNX session, and return P(Ray-Ban) via sigmoid. Preprocessing mirrors
 * the PyTorch eval transform (resize -> /255 -> ImageNet normalize). */
(function (root, factory) {
  var mod = factory(root);
  if (typeof module !== 'undefined' && module.exports) { module.exports = mod; }
  if (root) { root.DET = root.DET || {}; root.DET.cnn = mod; }
})(typeof self !== 'undefined' ? self : (typeof window !== 'undefined' ? window : null),
function (root) {
  'use strict';

  var _session = null;
  var _meta = null;   // {input, mean, std, threshold}

  function setSession(session, meta) { _session = session; _meta = meta; }
  function ready() { return !!_session; }
  function threshold() { return _meta ? _meta.threshold : 0.5; }

  /**
   * @param {cv.Mat} cropMat CV_8UC4 (RGBA) region crop
   * @param {object} cv
   * @returns {Promise<number>} P(Ray-Ban) in [0,1]
   */
  function infer(cropMat, cv) {
    var N = _meta.input, mean = _meta.mean, std = _meta.std;
    var resized = new cv.Mat();
    var interp = (cropMat.cols > N || cropMat.rows > N) ? cv.INTER_AREA : cv.INTER_LINEAR;
    cv.resize(cropMat, resized, new cv.Size(N, N), 0, 0, interp);
    var data = resized.data;                 // Uint8 RGBA, length N*N*4
    var plane = N * N;
    var f = new Float32Array(3 * plane);
    var i;
    for (i = 0; i < plane; i += 1) {
      f[i]           = ((data[i * 4]     / 255) - mean[0]) / std[0];  // R
      f[plane + i]   = ((data[i * 4 + 1] / 255) - mean[1]) / std[1];  // G
      f[2 * plane + i] = ((data[i * 4 + 2] / 255) - mean[2]) / std[2]; // B
    }
    resized.delete();
    var ort = root.ort;
    var tensor = new ort.Tensor('float32', f, [1, 3, N, N]);
    return _session.run({ input: tensor }).then(function (out) {
      var key = out.logit ? 'logit' : Object.keys(out)[0];
      var logit = out[key].data[0];
      return 1 / (1 + Math.exp(-logit));
    });
  }

  return { setSession: setSession, ready: ready, threshold: threshold, infer: infer };
});
