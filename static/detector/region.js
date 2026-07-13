/* Locate the glasses REGION for the CNN classifier. Port of pipeline/region.py.
 * Worn (face found): face box + eye/eyeglasses cascades fix the eye line, region
 * spans the face width across that band. Held/studio: padded segment() bbox.
 * Fallback: central 90%. Eye cascades are loaded once via loadEyes(). */
(function (root, factory) {
  var mod = factory(root);
  if (typeof module !== 'undefined' && module.exports) { module.exports = mod; }
  if (root) { root.DET = root.DET || {}; root.DET.region = mod; }
})(typeof self !== 'undefined' ? self : (typeof window !== 'undefined' ? window : null),
function (root) {
  'use strict';

  var _eye = [];        // [CascadeClassifier] eyeglasses, eye
  var _ready = false;

  function ready() { return _ready; }

  /** Load eye/eyeglasses cascades. bytes: {eyeglasses:Uint8Array, eye:Uint8Array} */
  function loadEyes(cv, bytes) {
    function mk(name, data) {
      try { cv.FS_unlink(name); } catch (e) { /* first time */ }
      cv.FS_createDataFile('/', name, data, true, false, false);
      var c = new cv.CascadeClassifier();
      c.load(name);
      return c;
    }
    _eye = [mk('eyeglasses.xml', bytes.eyeglasses), mk('eye.xml', bytes.eye)];
    _ready = true;
  }

  /** Median eye-centre y from cascades in the face's upper 62%, else a typical line. */
  function eyeLine(gray, face, cv) {
    var fallback = face.y + 0.38 * face.h;
    if (!_ready) { return fallback; }
    var rx = Math.max(0, face.x), ry = Math.max(0, face.y);
    var rw = Math.min(gray.cols - rx, face.w);
    var rh = Math.min(face.y + Math.floor(0.62 * face.h), gray.rows) - ry;
    if (rw <= 0 || rh <= 0) { return fallback; }
    var roi = gray.roi(new cv.Rect(rx, ry, rw, rh));
    var side = Math.max(12, Math.floor(0.12 * face.w));
    var ms = new cv.Size(side, side), empty = new cv.Size();
    var cys = [], i, j;
    for (i = 0; i < _eye.length; i += 1) {
      var rv = new cv.RectVector();
      _eye[i].detectMultiScale(roi, rv, 1.1, 4, 0, ms, empty);
      for (j = 0; j < rv.size(); j += 1) {
        var r = rv.get(j);
        cys.push(ry + r.y + r.height / 2);
      }
      rv.delete();
    }
    roi.delete();
    if (!cys.length) { return fallback; }
    cys.sort(function (a, b) { return a - b; });
    return cys[Math.floor(cys.length / 2)];
  }

  function clip(x0, y0, x1, y1, W, H) {
    x0 = Math.max(0, Math.round(x0)); y0 = Math.max(0, Math.round(y0));
    x1 = Math.min(W, Math.round(x1)); y1 = Math.min(H, Math.round(y1));
    return { x: x0, y: y0, w: Math.max(0, x1 - x0), h: Math.max(0, y1 - y0) };
  }

  /** @returns {{x,y,w,h,method}} region bbox in canonical pixels. */
  function locateRegion(pre, cfg, cv, face) {
    var H = pre.gray.rows, W = pre.gray.cols;
    var fd = root.DET && root.DET.facedet;
    if (face === undefined) {
      face = (fd && fd.ready && fd.ready()) ? fd.detectFace(pre.gray, cv) : null;
    }
    if (face) {
      var eyeCy = eyeLine(pre.gray, face, cv);
      var b = clip(face.x - cfg.region_wpad * face.w, eyeCy - cfg.region_up * face.h,
                   face.x + face.w + cfg.region_wpad * face.w, eyeCy + cfg.region_down * face.h, W, H);
      if (b.w > 20 && b.h > 20) { b.method = 'worn'; return b; }
    }
    var seg = root.DET.segment.segment(pre.grayEq, cfg, cv);
    var res;
    if (seg.valid && seg.method !== 'center-crop') {
      var bx = seg.bbox[0], by = seg.bbox[1], bw = seg.bbox[2], bh = seg.bbox[3];
      res = clip(bx - cfg.region_seg_pad * bw, by - cfg.region_seg_pad * bh,
                 bx + bw + cfg.region_seg_pad * bw, by + bh + cfg.region_seg_pad * bh, W, H);
      res.method = 'held';
    } else {
      res = clip(0.05 * W, 0.05 * H, 0.95 * W, 0.95 * H, W, H);
      res.method = 'center';
    }
    if (seg.free) { seg.free(); }
    return res;
  }

  return { loadEyes: loadEyes, ready: ready, locateRegion: locateRegion,
           eyeLine: eyeLine };
});
