/* YOLOX-Nano detector — the trained two-class glasses detector, run in the browser.
 *
 * This is the SAME model used server-side (runs/rayban_yolox/best_ckpt.pth,
 * exported to static/models/yolox_nano_rayban.onnx). It emits per-class boxes
 * (class 0 = rayban_meta, class 1 = glasses), so the verdict is three-way:
 *
 *   META (SMART GLASSES) = rayban box confidence  >= conf          (default 0.40)
 *   NORMAL               = else glasses box conf. >= conf_glasses  (default 0.35)
 *   NOGLASSES            = otherwise (no eyewear on the face)
 *
 * A single-class model file (meta.json without "classes") still works: the decode
 * falls back to one class and the verdict to the old binary META/NORMAL.
 *
 * There is no per-corner probability (unlike the older corner CNN); the meter
 * shows the rayban box confidence and the two corner cards read n/a. The overlay
 * draws the winning box, plus a camera marker at each top-outer end-piece (parity
 * with mark_cameras.py) when the verdict is META.
 *
 * Preprocess and decode are a bit-faithful port of YOLOX's own preproc +
 * demo_postprocess (verified against infer.YoloxDetector, 0.00px diff): letterbox
 * to NxN with 114 padding, BGR channel order, NO normalization; the ONNX output is
 * a RAW [1,8400,5+numClasses] grid (exported decode_in_inference=False) decoded
 * here with the stride grid. Geometry/thresholds come from
 * static/models/yolox.meta.json so the browser and training config can never
 * drift. */
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
    /* number of classes comes from meta.json ("classes": [...]); an old
       single-class meta has no list and decodes 6-wide rows as before */
    _meta.numClasses = (_meta.classes && _meta.classes.length) || 1;
    if (_meta.conf_glasses == null) { _meta.conf_glasses = 0.35; }
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
   * Decode the raw [8400, 5+numClasses] output and return the highest-scoring box
   * PER CLASS in ORIGINAL image pixels. Pure/testable: no cv or ORT here.
   * @param {Float32Array|number[]} d  flat 8400*(5+nc) [rx,ry,rw,rh,obj,cls0,cls1..]
   * @param {number} r                 letterbox ratio from preprocess()
   * @returns {Array<{x1:number,y1:number,x2:number,y2:number,score:number}>}
   *          index 0 = rayban_meta, index 1 = glasses (when present)
   */
  function decodePerClass(d, r) {
    var g = buildGrid(_meta.input, _meta.strides);
    var nc = _meta.numClasses || 1;
    var stride = 5 + nc;
    var rows = g.gx.length, i, c, base, obj, score;
    var best = [], bi = [];
    for (c = 0; c < nc; c += 1) { best.push(-1); bi.push(0); }
    for (i = 0; i < rows; i += 1) {
      base = i * stride;
      obj = d[base + 4];                       // obj & cls already sigmoid
      for (c = 0; c < nc; c += 1) {
        score = obj * d[base + 5 + c];
        if (score > best[c]) { best[c] = score; bi[c] = i; }
      }
    }
    var out = [];
    for (c = 0; c < nc; c += 1) {
      base = bi[c] * stride;
      var s = g.st[bi[c]];
      var cx = (d[base] + g.gx[bi[c]]) * s;
      var cy = (d[base + 1] + g.gy[bi[c]]) * s;
      var w = Math.exp(d[base + 2]) * s;
      var h = Math.exp(d[base + 3]) * s;
      out.push({
        x1: (cx - w / 2) / r, y1: (cy - h / 2) / r,
        x2: (cx + w / 2) / r, y2: (cy + h / 2) / r,
        score: best[c]
      });
    }
    return out;
  }

  /** Back-compat: single highest-scoring box across all classes. */
  function decodeTop(d, r) {
    var per = decodePerClass(d, r), top = per[0], i;
    for (i = 1; i < per.length; i += 1) {
      if (per[i].score > top.score) { top = per[i]; }
    }
    return top;
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
      var per = decodePerClass(out[key].data, pre.r);
      var rb = per[0];                       // class 0: rayban_meta
      var gl = per.length > 1 ? per[1] : null;  // class 1: glasses
      var conf = _meta.conf, confG = _meta.conf_glasses;
      var isMeta = rb.score >= conf;
      var isGlasses = !isMeta && !!gl && gl.score >= confG;
      /* single-class model (no glasses class): keep the old binary verdict */
      var verdict = isMeta ? 'META' : (isGlasses || !gl ? 'NORMAL' : 'NOGLASSES');
      var r3 = function (p) { return Math.round(p * 1000) / 1000; };

      var geom = { colorW: W, colorH: H };
      var src = isMeta ? rb : (isGlasses ? gl : null);
      if (src) {
        var x = Math.max(0, src.x1), y = Math.max(0, src.y1);
        var w = Math.min(W, src.x2) - x, h = Math.min(H, src.y2) - y;
        geom.region = [x, y, w, h];
        if (isMeta) { geom.cameras = cameraPoints(x, y, w, h); }
      }
      var reason;
      if (isMeta) {
        reason = 'Ray-Ban Meta glasses detected — box confidence ' +
          r3(rb.score) + ' (need >= ' + conf + ')';
      } else if (isGlasses) {
        reason = 'normal glasses detected — glasses confidence ' + r3(gl.score) +
          ' (need >= ' + confG + '); Ray-Ban Meta ' + r3(rb.score) +
          ' (need >= ' + conf + ')';
      } else if (gl) {
        reason = 'no eyewear detected — Ray-Ban Meta ' + r3(rb.score) +
          ' (need >= ' + conf + '), glasses ' + r3(gl.score) +
          ' (need >= ' + confG + ')';
      } else {
        reason = 'no Ray-Ban Meta glasses detected — box confidence ' +
          r3(rb.score) + ' (need >= ' + conf + ')';
      }
      return {
        verdict: verdict,
        per_corner: { L: { cam_prob: null }, R: { cam_prob: null } },
        prob_left: null, prob_right: null,
        overall_score: r3(rb.score),
        glasses_score: gl ? r3(gl.score) : null,
        threshold: conf,
        threshold_glasses: confG,
        reason: reason,
        segment_method: 'yolox',
        locate_method: 'yolox-nano',
        geom: geom
      };
    });
  }

  return { setSession: setSession, ready: ready, meta: meta,
           preprocess: preprocess, decodeTop: decodeTop,
           decodePerClass: decodePerClass, detect: detect,
           _buildGrid: buildGrid };
});
