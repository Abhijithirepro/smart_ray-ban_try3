/* Corner module detector — port of pipeline/corners.py + the three-tier verdict.
 *
 * The Ray-Ban Meta camera is a small module at each TOP-OUTER frame corner. This
 * module crops both corners from the FULL-RESOLUTION frame (the whole-face CNN
 * could not resolve the module after downscaling), classifies each with a CNN via
 * onnxruntime-web, and decides:
 *
 *   YES   (META)  = BOTH corners >= tc_hi   — straight-on, both modules seen
 *   MAYBE         = one corner  >= tc_maybe — profile / one side occluded
 *   NO    (NORMAL)= otherwise
 *
 * tc_hi is derived structurally in training (clears every normal image's
 * second-highest corner), so no normal glasses can ever reach YES. Face anchor:
 * DET.facedet (Haar); vertical anchor: DET.region.eyeLine (eye cascades). The
 * right corner is mirrored to the left's canonical orientation (parity with
 * training). All geometry comes from corner_clf.meta.json so train and inference
 * can never drift. */
(function (root, factory) {
  var mod = factory(root);
  if (typeof module !== 'undefined' && module.exports) { module.exports = mod; }
  if (root) { root.DET = root.DET || {}; root.DET.corners = mod; }
})(typeof self !== 'undefined' ? self : (typeof window !== 'undefined' ? window : null),
function (root) {
  'use strict';

  var _session = null;
  var _meta = null;   // {input, mean, std, tc_hi, tc_maybe, corner_size, corner_yc, corner_out}

  function setSession(session, meta) { _session = session; _meta = meta; }
  function ready() { return !!_session; }
  function meta() { return _meta; }

  /** One top-outer corner box in image pixels (mirror of corners._corner_box). */
  function cornerBox(face, side, eyeCy, W, H) {
    var w = Math.round(_meta.corner_size * face.w);
    var h = w;
    var cy = eyeCy - _meta.corner_yc * face.h;
    var top = Math.round(cy - h / 2);
    var x = (side === 'L')
      ? Math.round(face.x - _meta.corner_out * face.w)
      : Math.round(face.x + face.w + _meta.corner_out * face.w - w);
    x = Math.max(0, Math.min(W - 1, x));
    top = Math.max(0, Math.min(H - 1, top));
    w = Math.max(1, Math.min(W - x, w));
    h = Math.max(1, Math.min(H - top, h));
    return { x: x, y: top, w: w, h: h };
  }

  /** Fill 3*N*N floats of `out` starting at `off` from an RGBA NxN mat (NCHW). */
  function fillTensor(rgba, out, off) {
    var N = _meta.input, mean = _meta.mean, std = _meta.std;
    var data = rgba.data, plane = N * N, i;
    for (i = 0; i < plane; i += 1) {
      out[off + i] = ((data[i * 4] / 255) - mean[0]) / std[0];
      out[off + plane + i] = ((data[i * 4 + 1] / 255) - mean[1]) / std[1];
      out[off + 2 * plane + i] = ((data[i * 4 + 2] / 255) - mean[2]) / std[2];
    }
  }

  /** Three-tier verdict from the two corner probabilities. */
  function tier(pL, pR) {
    var hits = (pL >= _meta.tc_hi ? 1 : 0) + (pR >= _meta.tc_hi ? 1 : 0);
    if (hits >= 2) { return 'YES'; }
    if (pL >= _meta.tc_maybe || pR >= _meta.tc_maybe) { return 'MAYBE'; }
    return 'NO';
  }

  /**
   * Full corner pipeline on a FULL-RESOLUTION RGBA mat. Resolves to a payload
   * (renderResult-compatible), or null when no face is found (caller falls back
   * to the region CNN). Runs both corners through ORT in ONE batched call.
   * @param {cv.Mat} rgbaMat CV_8UC4, native resolution
   * @param {object} cv OpenCV.js module
   * @returns {Promise<object|null>}
   */
  function detect(rgbaMat, cv) {
    var facedet = root.DET.facedet, region = root.DET.region;
    if (!_session || !facedet || !facedet.ready()) { return Promise.resolve(null); }
    var W = rgbaMat.cols, H = rgbaMat.rows;
    var gray = new cv.Mat();
    cv.cvtColor(rgbaMat, gray, cv.COLOR_RGBA2GRAY);
    var face = facedet.detectFace(gray, cv);
    if (!face) { gray.delete(); return Promise.resolve(null); }
    var eyeCy = (region && region.ready && region.ready())
      ? region.eyeLine(gray, face, cv) : (face.y + 0.38 * face.h);
    gray.delete();

    var N = _meta.input;
    var boxes = { L: cornerBox(face, 'L', eyeCy, W, H),
                  R: cornerBox(face, 'R', eyeCy, W, H) };
    var f = new Float32Array(2 * 3 * N * N);
    var sides = ['L', 'R'], s, b, roi, flipped, resized, srcMat;
    for (s = 0; s < 2; s += 1) {
      b = boxes[sides[s]];
      roi = rgbaMat.roi(new cv.Rect(b.x, b.y, b.w, b.h));
      srcMat = roi;
      if (sides[s] === 'R') {           // mirror R to canonical orientation
        flipped = new cv.Mat();
        cv.flip(roi, flipped, 1);
        srcMat = flipped;
      }
      resized = new cv.Mat();
      var interp = (srcMat.cols > N || srcMat.rows > N) ? cv.INTER_AREA : cv.INTER_CUBIC;
      cv.resize(srcMat, resized, new cv.Size(N, N), 0, 0, interp);
      fillTensor(resized, f, s * 3 * N * N);
      resized.delete();
      if (srcMat !== roi) { srcMat.delete(); }
      roi.delete();
    }

    var ort = root.ort;
    var tensor = new ort.Tensor('float32', f, [2, 3, N, N]);
    return _session.run({ input: tensor }).then(function (out) {
      var key = out.logit ? 'logit' : Object.keys(out)[0];
      var d = out[key].data;
      var pL = 1 / (1 + Math.exp(-d[0]));
      var pR = 1 / (1 + Math.exp(-d[1]));
      var v = tier(pL, pR);
      var verdict = (v === 'YES') ? 'META' : (v === 'MAYBE' ? 'MAYBE' : 'NORMAL');
      var r3 = function (p) { return Math.round(p * 1000) / 1000; };
      var reason = {
        YES: 'camera module recognised in BOTH top-outer corners',
        MAYBE: 'a camera module may be present (one corner fired) — check manually',
        NO: 'no camera module found in either top-outer corner'
      }[v] + ' — Pcam L=' + r3(pL) + ' R=' + r3(pR);
      return {
        verdict: verdict,
        tier: v,
        prob_left: r3(pL),
        prob_right: r3(pR),
        per_corner: { L: { cam_prob: r3(pL) }, R: { cam_prob: r3(pR) } },
        overall_score: r3(Math.max(pL, pR)),
        threshold: _meta.tc_hi,
        threshold_maybe: _meta.tc_maybe,
        reason: reason,
        segment_method: 'corner:' + face.method,
        locate_method: 'corners:eyeline',
        geom: {
          colorW: W, colorH: H,
          face: [face.x, face.y, face.w, face.h],
          rois: [
            { side: 'L', x: boxes.L.x, y: boxes.L.y, w: boxes.L.w, h: boxes.L.h },
            { side: 'R', x: boxes.R.x, y: boxes.R.y, w: boxes.R.w, h: boxes.R.h }
          ]
        }
      };
    });
  }

  return { setSession: setSession, ready: ready, meta: meta,
           tier: tier, detect: detect };
});
