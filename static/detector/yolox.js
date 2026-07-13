/* YOLOX-Nano detector — the trained Ray-Ban Meta detector, run in the browser.
 *
 * This is the SAME model used server-side (runs/rayban_yolox/best_ckpt.pth,
 * exported to static/models/yolox_nano_rayban.onnx). It emits ONE whole-glasses
 * box + a confidence, so the verdict is simply:
 *
 *   META (SMART GLASSES) = top box confidence >= conf   (default 0.40)
 *   NORMAL               = otherwise
 *
 * There is no per-corner probability (unlike the older corner CNN); the meter
 * shows the box confidence and the two corner cards read n/a. The overlay draws
 * the glasses box plus a camera marker at each top-outer end-piece (parity with
 * mark_cameras.py), where the Ray-Ban Meta cameras physically sit.
 *
 * Preprocess and decode are a bit-faithful port of YOLOX's own preproc +
 * demo_postprocess (verified against infer.YoloxDetector, 0.00px diff): letterbox
 * to NxN with 114 padding, BGR channel order, NO normalization; the ONNX output is
 * a RAW [1,8400,6] grid (exported decode_in_inference=False) decoded here with the
 * stride grid. Geometry/thresholds come from static/models/yolox.meta.json so the
 * browser and the training config can never drift. */
(function (root, factory) {
  var mod = factory(root);
  if (typeof module !== 'undefined' && module.exports) { module.exports = mod; }
  if (root) { root.DET = root.DET || {}; root.DET.yolox = mod; }
})(typeof self !== 'undefined' ? self : (typeof window !== 'undefined' ? window : null),
function (root) {
  'use strict';

  var _session = null;
  var _meta = null;      // {input, strides, conf, nms, ...}
  var _grid = null;      // cached {gx, gy, st} Int32Arrays of length sum(size^2)

  function setSession(session, meta) {
    _session = session;
    _meta = meta || {};
    if (_meta.input == null) { _meta.input = 640; }
    if (!_meta.strides) { _meta.strides = [8, 16, 32]; }
    if (_meta.conf == null) { _meta.conf = 0.40; }
    _grid = null;         // rebuilt lazily for the current input/strides
  }
  function ready() { return !!_session; }
  function meta() { return _meta; }

  /** Build (and cache) the anchor-point grid: for each stride, row-major over
   *  (y,x) — exactly YOLOX demo_postprocess's meshgrid order. */
  function buildGrid(n, strides) {
    if (_grid && _grid.n === n && _grid.key === strides.join(',')) { return _grid; }
    var total = 0, si;
    for (si = 0; si < strides.length; si += 1) {
      var sz = Math.floor(n / strides[si]); total += sz * sz;
    }
    var gx = new Int32Array(total), gy = new Int32Array(total), st = new Int32Array(total);
    var k = 0;
    for (si = 0; si < strides.length; si += 1) {
      var s = strides[si], size = Math.floor(n / s), yi, xi;
      for (yi = 0; yi < size; yi += 1) {
        for (xi = 0; xi < size; xi += 1) {
          gx[k] = xi; gy[k] = yi; st[k] = s; k += 1;
        }
      }
    }
    _grid = { gx: gx, gy: gy, st: st, n: n, key: strides.join(',') };
    return _grid;
  }

  /**
   * Letterbox an RGBA cv.Mat to NxN (114 padding, top-left aligned) and pack it
   * into an NCHW float32 tensor in BGR plane order with NO normalization.
   * @returns {{data: Float32Array, r: number}}  r = resize ratio (orig->letterbox)
   */
  function preprocess(rgbaMat, cv) {
    var n = _meta.input, W = rgbaMat.cols, H = rgbaMat.rows;
    var r = Math.min(n / H, n / W);
    var nw = Math.floor(W * r), nh = Math.floor(H * r);
    var resized = new cv.Mat();
    cv.resize(rgbaMat, resized, new cv.Size(nw, nh), 0, 0, cv.INTER_LINEAR);
    var data = resized.data;                 // RGBA, length nw*nh*4
    var plane = n * n;
    var out = new Float32Array(3 * plane);
    var i;
    for (i = 0; i < out.length; i += 1) { out[i] = 114; }   // pad value
    var y, x, sidx, pos;
    for (y = 0; y < nh; y += 1) {
      for (x = 0; x < nw; x += 1) {
        sidx = (y * nw + x) * 4;
        pos = y * n + x;
        out[pos] = data[sidx + 2];             // B plane
        out[plane + pos] = data[sidx + 1];     // G plane
        out[2 * plane + pos] = data[sidx];     // R plane
      }
    }
    resized.delete();
    return { data: out, r: r };
  }

  /**
   * Decode the raw [8400,6] output and return the single highest-scoring box in
   * ORIGINAL image pixels. Pure/testable: no cv or ORT here.
   * @param {Float32Array|number[]} d  flat 8400*6 [rx,ry,rw,rh,obj,cls]
   * @param {number} r                 letterbox ratio from preprocess()
   * @returns {{x1:number,y1:number,x2:number,y2:number,score:number}}
   */
  function decodeTop(d, r) {
    var g = buildGrid(_meta.input, _meta.strides);
    var rows = g.gx.length, i, base, score, best = -1, bi = 0;
    for (i = 0; i < rows; i += 1) {
      base = i * 6;
      score = d[base + 4] * d[base + 5];       // obj * cls (both already sigmoid)
      if (score > best) { best = score; bi = i; }
    }
    base = bi * 6;
    var s = g.st[bi];
    var cx = (d[base] + g.gx[bi]) * s;
    var cy = (d[base + 1] + g.gy[bi]) * s;
    var w = Math.exp(d[base + 2]) * s;
    var h = Math.exp(d[base + 3]) * s;
    return {
      x1: (cx - w / 2) / r, y1: (cy - h / 2) / r,
      x2: (cx + w / 2) / r, y2: (cy + h / 2) / r,
      score: best
    };
  }

  /** Two camera-module points at the top-outer corners of a box (mark_cameras
   *  camera_points: inset 0.10*w, 0.13*h so the marker lands on the end-piece). */
  function cameraPoints(x, y, w, h) {
    var inx = 0.10 * w, iny = 0.13 * h;
    return [[x + inx, y + iny], [x + w - inx, y + iny]];
  }

  /**
   * Run the detector on a full-resolution RGBA cv.Mat. Always resolves to a
   * renderResult-compatible payload (verdict NORMAL when nothing clears conf).
   * @returns {Promise<object>}
   */
  function detect(rgbaMat, cv) {
    if (!_session) { return Promise.resolve(null); }
    var W = rgbaMat.cols, H = rgbaMat.rows;
    var rgb = new cv.Mat();
    cv.cvtColor(rgbaMat, rgb, cv.COLOR_RGBA2RGB);       // drop alpha; keep RGB order
    // preprocess reads channels explicitly, so RGB vs RGBA only changes the stride
    var rgba4 = new cv.Mat();
    cv.cvtColor(rgb, rgba4, cv.COLOR_RGB2RGBA);
    rgb.delete();
    var pre = preprocess(rgba4, cv);
    rgba4.delete();

    var ort = root.ort;
    var t = new ort.Tensor('float32', pre.data, [1, 3, _meta.input, _meta.input]);
    return _session.run({ images: t }).then(function (out) {
      var key = out.output ? 'output' : Object.keys(out)[0];
      var box = decodeTop(out[key].data, pre.r);
      var conf = _meta.conf;
      var isMeta = box.score >= conf;
      var r3 = function (p) { return Math.round(p * 1000) / 1000; };

      var geom = { colorW: W, colorH: H };
      if (isMeta) {
        var x = Math.max(0, box.x1), y = Math.max(0, box.y1);
        var w = Math.min(W, box.x2) - x, h = Math.min(H, box.y2) - y;
        geom.region = [x, y, w, h];
        geom.cameras = cameraPoints(x, y, w, h);
      }
      return {
        verdict: isMeta ? 'META' : 'NORMAL',
        per_corner: { L: { cam_prob: null }, R: { cam_prob: null } },
        prob_left: null, prob_right: null,
        overall_score: r3(box.score),
        threshold: conf,
        reason: (isMeta ? 'Ray-Ban Meta glasses detected' : 'no Ray-Ban Meta glasses detected') +
          ' — box confidence ' + r3(box.score) + ' (need >= ' + conf + ')',
        segment_method: 'yolox',
        locate_method: 'yolox-nano',
        geom: geom
      };
    });
  }

  return { setSession: setSession, ready: ready, meta: meta,
           preprocess: preprocess, decodeTop: decodeTop, detect: detect,
           _buildGrid: buildGrid };
});
