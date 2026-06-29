/* MODULE SCAN — front-end controller (ES5, vanilla).
   Handles upload (click + drag/drop), posts to /api/detect, renders the
   verdict, confidence meter, per-corner scores, signature chips and overlay. */
(function () {
  'use strict';

  var ENDPOINT = '/api/detect';

  /* Signature features shown as chips, in display order. */
  var FEATURES = [
    { key: 'f_circle', label: 'BEZEL' },
    { key: 'f_spec',   label: 'GLINT' },
    { key: 'f_dark',   label: 'DARK CORE' },
    { key: 'f_thick',  label: 'CHUNKY' }
  ];

  /**
   * Grab an element by id (null-safe helper).
   * @param {string} id
   * @returns {HTMLElement|null}
   */
  function el(id) {
    return document.getElementById(id);
  }

  /**
   * Show a status line, optionally flagged as an error or busy state.
   * @param {string} msg
   * @param {string} mode  '' | 'error' | 'busy'
   */
  function setStatus(msg, mode) {
    var s = el('status');
    if (!s) { return; }
    s.textContent = msg;
    s.className = 'status' + (mode ? ' is-' + mode : '');
  }

  /**
   * Format a 0..1 number to two decimals.
   * @param {number} n
   * @returns {string}
   */
  function fmt(n) {
    return (Math.round(n * 100) / 100).toFixed(2);
  }

  /**
   * Pick the stronger corner's feature dict (the corner that drives the call).
   * @param {Object} payload
   * @returns {Object}
   */
  function firingFeatures(payload) {
    var pf = payload.per_feature || {};
    if ((payload.score_right || 0) > (payload.score_left || 0)) {
      return pf.R || {};
    }
    return pf.L || {};
  }

  /**
   * Render the signature chips for the firing corner.
   * @param {Object} feats
   */
  function renderChips(feats) {
    var wrap = el('feat-chips');
    var i, f, val, chip, html;
    if (!wrap) { return; }
    html = '';
    for (i = 0; i < FEATURES.length; i++) {
      f = FEATURES[i];
      val = feats[f.key] || 0;
      chip = val >= 0.5 ? 'chip on' : 'chip';
      html += '<span class="' + chip + '">' + f.label + '</span>';
    }
    wrap.innerHTML = html;
  }

  /**
   * Paint the full result panel from an API payload.
   * @param {Object} payload
   */
  function renderResult(payload) {
    var result = el('result');
    var isMeta = payload.verdict === 'META';
    var conf = payload.overall_score || 0;
    var threshold = payload.threshold || 0.6;

    el('overlay').src = payload.overlay || '';
    el('verdict-text').textContent = isMeta ? 'SMART GLASSES' : 'NORMAL GLASSES';
    el('verdict-reason').textContent = isMeta
      ? 'camera module found in corner ' + (payload.fired_corner || '?')
      : (payload.reason || 'no camera module detected');

    el('conf-val').textContent = fmt(conf);
    el('score-l').textContent = fmt(payload.score_left || 0);
    el('score-r').textContent = fmt(payload.score_right || 0);

    renderChips(firingFeatures(payload));

    result.className = 'result ' + (isMeta ? 'is-meta' : 'is-normal');
    result.hidden = false;
    document.body.className = 'has-result';

    /* run the scan reveal, then fill the meter once layout is settled */
    void result.offsetWidth;
    result.className += ' is-scanning';
    window.setTimeout(function () {
      el('conf-fill').style.width = (conf * 100) + '%';
      el('conf-mark').style.left = (threshold * 100) + '%';
    }, 120);
    window.setTimeout(function () {
      result.className = 'result ' + (isMeta ? 'is-meta' : 'is-normal');
    }, 1100);

    setStatus('scan complete · ' + payload.segment_method + ' segmentation', '');
    result.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  /**
   * Upload an image file to the detector and render the response.
   * @param {File} file
   */
  function scan(file) {
    var xhr, form;
    if (!file) { return; }
    if (file.type && file.type.indexOf('image/') !== 0) {
      setStatus('that is not an image file', 'error');
      return;
    }
    setStatus('scanning corners for a camera module', 'busy');

    form = new FormData();
    form.append('image', file);

    xhr = new XMLHttpRequest();
    xhr.open('POST', ENDPOINT, true);
    xhr.onreadystatechange = function () {
      var data;
      if (xhr.readyState !== 4) { return; }
      try {
        data = JSON.parse(xhr.responseText);
      } catch (e) {
        setStatus('unexpected server response', 'error');
        return;
      }
      if (xhr.status === 200) {
        renderResult(data);
      } else {
        setStatus('error: ' + (data && data.error ? data.error : xhr.status), 'error');
      }
    };
    xhr.send(form);
  }

  /**
   * Wire up the drop zone, file input and reset button.
   */
  function init() {
    var drop = el('drop');
    var input = el('file');
    var again = el('again');

    if (input) {
      input.addEventListener('change', function () {
        if (input.files && input.files[0]) { scan(input.files[0]); }
      });
    }

    if (drop) {
      drop.addEventListener('keydown', function (ev) {
        if ((ev.key === 'Enter' || ev.key === ' ') && input) { input.click(); }
      });
      drop.addEventListener('dragover', function (ev) {
        ev.preventDefault();
        drop.className = 'drop is-drag';
      });
      drop.addEventListener('dragleave', function () {
        drop.className = 'drop';
      });
      drop.addEventListener('drop', function (ev) {
        ev.preventDefault();
        drop.className = 'drop';
        if (ev.dataTransfer && ev.dataTransfer.files && ev.dataTransfer.files[0]) {
          scan(ev.dataTransfer.files[0]);
        }
      });
    }

    if (again) {
      again.addEventListener('click', function () {
        var result = el('result');
        if (result) { result.hidden = true; }
        document.body.className = '';
        if (input) { input.value = ''; }
        setStatus('', '');
        window.scrollTo({ top: 0, behavior: 'smooth' });
      });
    }
  }

  document.addEventListener('DOMContentLoaded', init);
})();
