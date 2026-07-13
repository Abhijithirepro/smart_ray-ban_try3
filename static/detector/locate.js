/* Stage 3 - locate the two top-outer corner ROIs the classifier scores.
 * Port of pipeline/locate.py. Two paths (whichever yields scorable corners wins):
 *
 *   FACE path (worn glasses): a Haar face box anchors a small grid of candidate
 *     corner crops per side; features scores them all and keeps the peak P(camera).
 *   HELD path (glasses held / studio): the lens-hole method - split the frame bbox
 *     at its centre, take the largest interior hole per half as the lens, and anchor
 *     one corner ROI outside each lens on the end piece.
 *
 * locate() returns candidates:{L:[...],R:[...]}; the caller peaks per side.
 * twoLensGate() is the domain gate: a face was anchored, or two lenses were found.
 */
(function (root, factory) {
  var mod = factory();
  if (typeof module !== 'undefined' && module.exports) { module.exports = mod; }
  if (root) { root.DET = root.DET || {}; root.DET.locate = mod; }
})(typeof self !== 'undefined' ? self : (typeof window !== 'undefined' ? window : null),
function () {
  'use strict';

  // ------------------------------------------------------------------ //
  // FACE path (worn glasses)
  // ------------------------------------------------------------------ //
  function clipRoi(x, y, w, h, side, H, W) {
    var x0 = Math.max(0, Math.round(x)), y0 = Math.max(0, Math.round(y));
    var x1 = Math.min(W, Math.round(x + w)), y1 = Math.min(H, Math.round(y + h));
    return {
      side: side, x: x0, y: y0, w: Math.max(0, x1 - x0), h: Math.max(0, y1 - y0),
      camCx: (x0 + x1) / 2, camCy: (y0 + y1) / 2
    };
  }

  /** Grid of candidate corner crops in one top-outer region of the face box. */
  function faceCandidates(face, side, cfg, H, W) {
    var out = [];
    var si, di, dj;
    for (si = 0; si < cfg.face_search_sizes.length; si += 1) {
      var w = face.w * cfg.face_search_sizes[si];
      var h = w;
      for (di = 0; di < cfg.face_search_dy.length; di += 1) {
        var cy = face.y + face.h * cfg.face_search_dy[di];
        for (dj = 0; dj < cfg.face_search_dx.length; dj += 1) {
          var dx = cfg.face_search_dx[dj];
          var x = side === 'L'
            ? face.x + face.w * dx
            : face.x + face.w - w - face.w * dx;
          out.push(clipRoi(x, cy - h / 2, w, h, side, H, W));
        }
      }
    }
    return out;
  }

  function locateFace(face, cfg, H, W) {
    var cand = { L: faceCandidates(face, 'L', cfg, H, W),
                 R: faceCandidates(face, 'R', cfg, H, W) };
    return {
      path: 'face', candidates: cand, method: 'haar:' + face.method, face: face,
      lensLeft: null, lensRight: null, rLens: 0,
      rois: [cand.L[0], cand.R[0]]   // placeholder; caller replaces with peak
    };
  }

  // ------------------------------------------------------------------ //
  // HELD path (glasses held up / studio) - lens-hole method
  // ------------------------------------------------------------------ //
  /** Interior holes (lens openings) of the frame contour, in full-image px. */
  function holesInRegion(grayEq, seg, cv) {
    var x = seg.bbox[0], y = seg.bbox[1], w = seg.bbox[2], h = seg.bbox[3];
    var crop = grayEq.roi(new cv.Rect(x, y, w, h));
    var th = new cv.Mat();
    cv.threshold(crop, th, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU);
    var k5 = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(5, 5));
    cv.morphologyEx(th, th, cv.MORPH_CLOSE, k5, new cv.Point(-1, -1), 1,
      cv.BORDER_CONSTANT, cv.morphologyDefaultBorderValue());
    k5.delete();

    var contours = new cv.MatVector();
    var hier = new cv.Mat();
    cv.findContours(th, contours, hier, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE);
    var holes = [];
    var par = hier.data32S; // [next,prev,child,parent] per contour
    var i;
    for (i = 0; i < contours.size(); i += 1) {
      if (par[i * 4 + 3] === -1) { continue; } // outer contour -> skip
      var area = cv.contourArea(contours.get(i));
      if (area < 0.02 * w * h) { continue; }
      var b = cv.boundingRect(contours.get(i));
      holes.push([x + b.x + b.width / 2, y + b.y + b.height / 2, b.width, b.height, area]);
    }
    contours.delete(); hier.delete(); th.delete(); crop.delete();
    return holes;
  }

  function lensFromHole(hole) {
    return { cx: hole[0], cy: hole[1], hw: hole[2] / 2, hh: hole[3] / 2 };
  }

  function fallbackLens(seg, side, cv) {
    var x = seg.bbox[0], y = seg.bbox[1], w = seg.bbox[2], h = seg.bbox[3];
    var halfW = side === 'L' ? Math.floor(w / 2) : w - Math.floor(w / 2);
    var halfX = side === 'L' ? x : x + Math.floor(w / 2);
    var half = seg.frameMask.roi(new cv.Rect(halfX, y, halfW, h));
    var m = cv.moments(half, false);
    half.delete();
    var cx, cy;
    if (m.m00 > 0) { cx = m.m10 / m.m00; cy = m.m01 / m.m00; }
    else { cx = w * 0.25; cy = h * 0.5; }
    var baseX = side === 'L' ? x : x + Math.floor(w / 2);
    return { cx: baseX + cx, cy: y + cy, hw: 0.22 * w, hh: 0.50 * h };
  }

  /* Search region OUTSIDE the lens, on the end piece, where the camera lives.
     Capped at roi_outer_max * lens.hw beyond the lens edge so it can't slide onto
     a hand/hair when the frame bbox is large (port of _make_roi). */
  function makeRoi(lens, bbox, side, cfg, H, W) {
    var bx = bbox[0], by = bbox[1], bw = bbox[2];
    var overlap = cfg.roi_lens_overlap * lens.hw;
    var outerCap = cfg.roi_outer_max * lens.hw;
    var x0, x1;
    if (side === 'L') {
      x0 = Math.max(bx, lens.cx - lens.hw - outerCap);
      x1 = lens.cx - lens.hw + overlap;
    } else {
      x0 = lens.cx + lens.hw - overlap;
      x1 = Math.min(bx + bw, lens.cx + lens.hw + outerCap);
    }
    var y0 = by;
    var y1 = lens.cy + cfg.roi_y_below * lens.hh;

    var minW = cfg.roi_min_w_frac * bw;
    if (x1 - x0 < minW) {
      if (side === 'L') { x1 = x0 + minW; } else { x0 = x1 - minW; }
    }
    x0 = Math.max(0, Math.round(x0)); y0 = Math.max(0, Math.round(y0));
    x1 = Math.min(W, Math.round(x1)); y1 = Math.min(H, Math.round(y1));
    return {
      side: side, x: x0, y: y0, w: x1 - x0, h: y1 - y0,
      camCx: (x0 + x1) / 2, camCy: (y0 + y1) / 2
    };
  }

  function locateHeld(grayEq, seg, cfg, cv) {
    var H = grayEq.rows, W = grayEq.cols;
    var x = seg.bbox[0], w = seg.bbox[2];
    var cxSplit = x + w / 2;
    var method = 'holes';

    var holes = holesInRegion(grayEq, seg, cv);
    var left = holes.filter(function (z) { return z[0] < cxSplit; });
    var right = holes.filter(function (z) { return z[0] >= cxSplit; });

    function maxByArea(arr) {
      return arr.reduce(function (a, b) { return b[4] > a[4] ? b : a; });
    }

    var lensL, lensR;
    if (left.length) { lensL = lensFromHole(maxByArea(left)); }
    else { lensL = fallbackLens(seg, 'L', cv); method = 'holes+fallback'; }
    if (right.length) { lensR = lensFromHole(maxByArea(right)); }
    else { lensR = fallbackLens(seg, 'R', cv); method = 'holes+fallback'; }

    var rLens = (lensL.hw + lensR.hw) / 2;
    var roiL = makeRoi(lensL, seg.bbox, 'L', cfg, H, W);
    var roiR = makeRoi(lensR, seg.bbox, 'R', cfg, H, W);
    return {
      path: 'held', candidates: { L: [roiL], R: [roiR] }, method: method,
      face: null, lensLeft: lensL, lensRight: lensR, rLens: rLens,
      rois: [roiL, roiR]
    };
  }

  /**
   * Face box present -> worn path; else held-glasses lens path.
   * @param {cv.Mat} grayEq CV_8UC1
   * @param {object} seg  segment() result
   * @param {object} cfg
   * @param {object} cv
   * @param {object|null} [face]  facedet.detectFace() result
   */
  function locate(grayEq, seg, cfg, cv, face) {
    if (face) { return locateFace(face, cfg, grayEq.rows, grayEq.cols); }
    return locateHeld(grayEq, seg, cfg, cv);
  }

  /**
   * Domain gate: confirm structure is present. Face anchored, or two lenses found.
   * @returns {{ok:boolean, reason:string}}
   */
  function twoLensGate(loc) {
    if (loc.path === 'face') {
      return { ok: !!loc.face, reason: loc.face ? 'face anchor' : 'no face' };
    }
    if (!loc.lensLeft || !loc.lensRight) {
      return { ok: false, reason: 'no two lenses found' };
    }
    return { ok: true, reason: 'held glasses frame' };
  }

  return { locate: locate, twoLensGate: twoLensGate };
});
