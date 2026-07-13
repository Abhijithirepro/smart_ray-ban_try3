/* Stage 1.5 - Haar face detection, the localization anchor for WORN glasses.
 * Port of pipeline/facedet.py.
 *
 * Pure classical CV can't find glasses on a face (hair/clothing/background
 * dominate), so we anchor on the *face* via OpenCV.js Haar cascades (Viola-Jones,
 * no neural net). Frontal (default + alt2), then a profile fallback. `locate`
 * derives the two top-outer corner ROIs from this box; no face -> held path.
 *
 * Cascades are loaded once into the OpenCV.js virtual FS by load(); the caller
 * supplies the XML bytes (fetch() in the browser, readFileSync in Node), keeping
 * this module free of any I/O specifics. Works under <script> (window.DET.facedet)
 * and Node (module.exports).
 */
(function (root, factory) {
  var mod = factory();
  if (typeof module !== 'undefined' && module.exports) { module.exports = mod; }
  if (root) { root.DET = root.DET || {}; root.DET.facedet = mod; }
})(typeof self !== 'undefined' ? self : (typeof window !== 'undefined' ? window : null),
function () {
  'use strict';

  /* Operating point mirrors facedet.py: upright only (mn=6, no rotation, +profile
     fallback) — tuned to give ZERO false faces on normal_frame while still
     catching worn faces. Adding rotated passes hallucinated faces on studio shots. */
  var SCALE_FACTOR = 1.05;
  var MIN_NEIGHBORS = 6;
  var MIN_SIZE_FRAC = 0.10;   // min face side as a fraction of min(H, W)

  var _cv = null;
  var _frontals = [];         // [CascadeClassifier] default, then alt2
  var _profile = null;        // CascadeClassifier
  var _ready = false;

  /** True once the cascades are loaded and detectFace() can run. */
  function ready() { return _ready; }

  /**
   * Load the three cascades into OpenCV.js. Call once, after cv is initialised.
   * @param {object} cv OpenCV.js module
   * @param {{frontalDefault:Uint8Array, frontalAlt2:Uint8Array, profile:Uint8Array}} bytes
   */
  function load(cv, bytes) {
    _cv = cv;
    function mk(name, data) {
      try { cv.FS_unlink(name); } catch (e) { /* not present yet */ }
      cv.FS_createDataFile('/', name, data, true, false, false);
      var c = new cv.CascadeClassifier();
      c.load(name);
      return c;
    }
    _frontals = [
      mk('fd_default.xml', bytes.frontalDefault),
      mk('fd_alt2.xml', bytes.frontalAlt2)
    ];
    _profile = mk('profileface.xml', bytes.profile);
    _ready = true;
  }

  /**
   * Largest plausible face box, or null. gray = canonical raw grayscale cv.Mat.
   * @param {cv.Mat} gray CV_8UC1
   * @param {object} [cv] OpenCV.js module (defaults to the one passed to load())
   * @returns {{x,y,w,h,method}|null}
   */
  function detectFace(gray, cv) {
    cv = cv || _cv;
    if (!_ready || !cv) { return null; }
    var H = gray.rows, W = gray.cols;
    var minSide = Math.floor(MIN_SIZE_FRAC * Math.min(H, W));
    var ms = new cv.Size(minSide, minSide);
    var maxs = new cv.Size();
    var best = null;   // {area, box:[x,y,w,h], method}

    function consider(box, method) {
      var area = box[2] * box[3];
      if (best === null || area > best.area) { best = { area: area, box: box, method: method }; }
    }

    function runOn(mat, cascade, tag, flipBack) {
      var rv = new cv.RectVector();
      cascade.detectMultiScale(mat, rv, SCALE_FACTOR, MIN_NEIGHBORS, 0, ms, maxs);
      var i;
      for (i = 0; i < rv.size(); i += 1) {
        var r = rv.get(i);
        var x = r.x;
        if (flipBack) { x = W - r.x - r.width; }
        consider([x, r.y, r.width, r.height], tag);
      }
      rv.delete();
    }

    // frontal cascades, upright
    var names = ['default', 'alt2'];
    var k;
    for (k = 0; k < _frontals.length; k += 1) {
      runOn(gray, _frontals[k], names[k] + '@0', false);
    }

    // profile fallback (image + mirror) only if nothing found yet
    if (best === null) {
      runOn(gray, _profile, 'profileL', false);
      var flipped = new cv.Mat();
      cv.flip(gray, flipped, 1);
      runOn(flipped, _profile, 'profileR', true);
      flipped.delete();
    }

    if (best === null) { return null; }
    var b = best.box;
    return { x: b[0], y: b[1], w: b[2], h: b[3], method: best.method };
  }

  return { load: load, ready: ready, detectFace: detectFace };
});
