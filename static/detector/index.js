/* Detector orchestrator — runs the whole in-browser pipeline and returns the
 * same payload shape the Flask /api/detect endpoint produced (viz.features_to_dict),
 * so static/app.js renders it unchanged.
 *
 *   preprocess -> (face?) -> segment -> locate -> corner_crop -> HOG -> logistic -> decide
 *
 * Worn glasses are anchored on a Haar face box (facedet); when no face is found we
 * fall back to the held-glasses lens path. Per side, every located candidate ROI is
 * scored and the PEAK P(camera) is kept (absorbs face-anchor placement imprecision).
 *
 * detectMat() takes a prebuilt RGBA cv.Mat (the caller owns/builds it from a
 * canvas in the browser, or from raw bytes in the Node parity harness).
 */
(function (root, factory) {
  var mod = factory(root);
  if (typeof module !== 'undefined' && module.exports) { module.exports = mod; }
  if (root) { root.DET = root.DET || {}; root.DET.detectMat = mod.detectMat; root.DET.detectRegion = mod.detectRegion; }
})(typeof self !== 'undefined' ? self : (typeof window !== 'undefined' ? window : null),
function (root) {
  'use strict';

  function dep(name) {
    if (root && root.DET && root.DET[name]) { return root.DET[name]; }
    if (typeof require !== 'undefined') { return require('./' + name + '.js'); }
    throw new Error('missing detector dependency: ' + name);
  }

  /* Like dep() but returns null instead of throwing — for optional modules
     (facedet) that may be absent or not yet loaded. */
  function optDep(name) {
    if (root && root.DET && root.DET[name]) { return root.DET[name]; }
    return null;
  }

  function round3(p) { return Math.round(p * 1000) / 1000; }

  /**
   * @param {cv.Mat} rgbaMat CV_8UC4 source image
   * @param {object} cv OpenCV.js module
   * @param {object} model {w,b,mu,sd,thresh}
   * @param {object} [cfg]  defaults to DET.config
   * @returns {object} payload
   */
  function detectMat(rgbaMat, cv, model, cfg) {
    var config = cfg || dep('config');
    var preprocess = dep('preprocess').preprocess;
    var segment = dep('segment').segment;
    var locateMod = dep('locate');
    var clf = dep('clf');
    var decide = dep('decide').decide;
    var facedet = optDep('facedet');   // optional: worn-glasses face anchor

    var pre = preprocess(rgbaMat, config, cv);
    /* Worn glasses: anchor on the face (Haar, raw gray). null -> held lens path. */
    var face = (facedet && facedet.ready && facedet.ready())
      ? facedet.detectFace(pre.gray, cv) : null;
    var seg = segment(pre.grayEq, config, cv);
    var loc = locateMod.locate(pre.grayEq, seg, config, cv, face);

    /* score EVERY candidate per side, keep the peak P(camera) and its ROI */
    var per = { L: { cam_prob: null }, R: { cam_prob: null } };
    var bestRoi = { L: null, R: null };
    var sides = ['L', 'R'], s, side, cands, j, p, best;
    for (s = 0; s < sides.length; s += 1) {
      side = sides[s];
      cands = (loc.candidates && loc.candidates[side]) || [];
      best = -1;
      for (j = 0; j < cands.length; j += 1) {
        p = clf.probFromCrop(clf.cornerCrop(pre.gray, cands[j], cv), model);
        if (p > best) { best = p; bestRoi[side] = cands[j]; }
      }
      if (cands.length) { per[side] = { cam_prob: round3(best) }; }
    }
    /* the best-scoring ROI per side drives the overlay */
    loc.rois = [bestRoi.L || (loc.candidates.L && loc.candidates.L[0]),
                bestRoi.R || (loc.candidates.R && loc.candidates.R[0])];

    var gate = locateMod.twoLensGate(loc);
    var verdict = decide(per.L.cam_prob, per.R.cam_prob, gate.ok, gate.reason, config);

    var payload = {
      verdict: verdict.verdict,
      prob_left: verdict.prob_left,
      prob_right: verdict.prob_right,
      fired_corner: verdict.fired_corner,
      reason: verdict.reason,
      r_lens: Math.round(loc.rLens * 100) / 100,
      bbox: seg.bbox.slice(),
      segment_method: seg.method,
      locate_method: loc.method,
      lens_centers: {
        L: loc.lensLeft ? [Math.round(loc.lensLeft.cx * 10) / 10, Math.round(loc.lensLeft.cy * 10) / 10] : null,
        R: loc.lensRight ? [Math.round(loc.lensRight.cx * 10) / 10, Math.round(loc.lensRight.cy * 10) / 10] : null
      },
      per_corner: per,
      threshold: config.cam_clf_thresh,
      // geometry for the optional in-browser overlay (app.js draws it)
      geom: { bbox: seg.bbox.slice(), rois: loc.rois, lensL: loc.lensLeft, lensR: loc.lensRight,
              face: loc.face || null, colorW: pre.color.cols, colorH: pre.color.rows }
    };

    seg.free();
    pre.free();
    return payload;
  }

  /**
   * CNN path: locate the glasses region, crop it, and classify Ray-Ban-vs-normal
   * with the whole-region MobileNetV3 (via onnxruntime-web). Returns a payload in
   * the same shape renderResult expects (per-corner probs are null; the single
   * region probability drives the meter). Async — ORT inference is a Promise.
   * @returns {Promise<object>}
   */
  function detectRegion(rgbaMat, cv, cfg) {
    var config = cfg || dep('config');
    var preprocess = dep('preprocess').preprocess;
    var region = dep('region');
    var cnn = dep('cnn');

    var pre = preprocess(rgbaMat, config, cv);
    var reg = region.locateRegion(pre, config, cv);
    var crop = pre.color.roi(new cv.Rect(reg.x, reg.y, reg.w, reg.h));
    return cnn.infer(crop, cv).then(function (prob) {
      crop.delete();
      var thr = cnn.threshold();
      var isMeta = prob >= thr;
      var pctP = Math.round(prob * 100) / 100;
      var payload = {
        verdict: isMeta ? 'META' : 'NORMAL',
        per_corner: { L: { cam_prob: null }, R: { cam_prob: null } },
        prob_left: null, prob_right: null,
        overall_score: prob,
        threshold: thr,
        reason: (isMeta ? 'camera glasses recognised' : 'no camera module found') +
          ' — P=' + pctP + ' over the located glasses region (' + reg.method + ')',
        segment_method: reg.method,
        locate_method: 'region:' + reg.method,
        geom: { region: [reg.x, reg.y, reg.w, reg.h],
                colorW: pre.color.cols, colorH: pre.color.rows }
      };
      pre.free();
      return payload;
    }, function (err) {
      pre.free();
      throw err;
    });
  }

  return { detectMat: detectMat, detectRegion: detectRegion };
});
