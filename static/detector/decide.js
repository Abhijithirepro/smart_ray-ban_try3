/* Stage 5 - verdict from the learned per-corner camera probability.
 * Port of pipeline/decide.py: META iff P(camera) >= thresh in BOTH corners, after
 * a domain gate confirms a glasses frame (face anchored, or a two-lens held frame).
 * The gate no longer rejects faces - detecting Meta worn on a face is a goal - so
 * the META/NORMAL split rests on the classifier.
 */
(function (root, factory) {
  var mod = factory();
  if (typeof module !== 'undefined' && module.exports) { module.exports = mod; }
  if (root) { root.DET = root.DET || {}; root.DET.decide = mod; }
})(typeof self !== 'undefined' ? self : (typeof window !== 'undefined' ? window : null),
function () {
  'use strict';

  function f2(p) { return p.toFixed(2); }

  /**
   * @param {number|null} pL peak P(camera), left corner
   * @param {number|null} pR peak P(camera), right corner
   * @param {boolean} gateOk   twoLensGate().ok
   * @param {string} gateReason
   * @param {object} cfg
   * @returns {{verdict, prob_left, prob_right, fired_corner, reason}}
   */
  function decide(pL, pR, gateOk, gateReason, cfg) {
    if (!gateOk) {
      return {
        verdict: 'NORMAL', prob_left: pL, prob_right: pR, fired_corner: null,
        reason: 'no glasses frame found (' + gateReason + ')'
      };
    }
    var thr = cfg.cam_clf_thresh;
    var hasL = pL >= thr, hasR = pR >= thr;
    var cue = 'P(cam) L=' + f2(pL) + ' R=' + f2(pR);
    var isMeta = hasL && hasR;
    var reason, fired;
    if (isMeta) { reason = 'camera in both corners [' + cue + ']'; fired = 'L+R'; }
    else if (hasL || hasR) {
      var only = hasL ? 'L' : 'R';
      reason = 'camera only in corner ' + only + ', need both [' + cue + ']';
      fired = null;
    } else { reason = 'no camera in either corner [' + cue + ']'; fired = null; }

    return {
      verdict: isMeta ? 'META' : 'NORMAL',
      prob_left: pL, prob_right: pR, fired_corner: fired, reason: reason
    };
  }

  return { decide: decide };
});
