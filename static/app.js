/* MODULE SCAN — front-end controller (ES5, vanilla).
   Runs the ENTIRE detection pipeline in the browser (OpenCV.js + the detector
   modules in static/detector/), with no server. Handles upload (click +
   drag/drop) and live camera, renders the verdict, confidence meter, per-corner
   P(camera) and an annotated overlay drawn on a canvas. */
(function () {
  'use strict';

  /* current input mode: 'camera' (default) or 'upload' */
  var mode = 'camera';

  /* the legacy HOG classifier weights ({w,b,mu,sd,thresh}); fallback only */
  var model = null;
  /* OpenCV.js runtime readiness */
  var cvReady = false;
  /* Haar cascades (face + eye): settled once loaded OR confirmed unavailable */
  var facesReady = false;
  /* whole-region CNN (onnxruntime-web): fallback classifier */
  var cnnReady = false;
  /* corner module CNN (onnxruntime-web): the older per-corner classifier — kept as
     a fallback if the YOLOX model fails to load */
  var cornersReady = false;
  /* YOLOX-Nano whole-glasses detector (onnxruntime-web): the PRIMARY detector —
     the same model trained/run server-side (box + confidence -> META/NORMAL) */
  var yoloxReady = false;

  /** True once the detector can run. YOLOX needs only cv + its ONNX session (no
      Haar cascades); the legacy pipeline additionally needs the cascades. */
  function detectReady() {
    if (cvReady && yoloxReady) { return true; }
    return cvReady && facesReady && (cornersReady || cnnReady || model !== null);
  }

  /** fetch a file as a Uint8Array (rejects on non-200). */
  function fetchBytes(url) {
    return fetch(url).then(function (r) {
      if (!r.ok) { throw new Error('HTTP ' + r.status + ' for ' + url); }
      return r.arrayBuffer();
    }).then(function (buf) { return new Uint8Array(buf); });
  }

  /**
   * Load the Haar cascades: 3 face (facedet) + 2 eye/eyeglasses (region). On any
   * failure we still mark "ready" so detection degrades gracefully.
   */
  function loadFaceCascades(cv) {
    var facedet = window.DET && window.DET.facedet;
    var region = window.DET && window.DET.region;
    if (!facedet) { facesReady = true; return; }
    var base = 'static/haar/';
    Promise.all([
      fetchBytes(base + 'haarcascade_frontalface_default.xml'),
      fetchBytes(base + 'haarcascade_frontalface_alt2.xml'),
      fetchBytes(base + 'haarcascade_profileface.xml'),
      fetchBytes(base + 'haarcascade_eye_tree_eyeglasses.xml'),
      fetchBytes(base + 'haarcascade_eye.xml')
    ]).then(function (b) {
      facedet.load(cv, { frontalDefault: b[0], frontalAlt2: b[1], profile: b[2] });
      if (region) { region.loadEyes(cv, { eyeglasses: b[3], eye: b[4] }); }
      facesReady = true;
      if (detectReady()) { setStatus('', ''); }
    }).catch(function (e) {
      if (window.console) { window.console.warn('cascades unavailable:', e); }
      facesReady = true;
      if (detectReady()) { setStatus('', ''); }
    });
  }

  /**
   * Load the whole-region CNN (ONNX) and its metadata via onnxruntime-web. On
   * failure the detector falls back to the HOG corner classifier.
   */
  function loadCnnModel() {
    var cnn = window.DET && window.DET.cnn;
    if (!cnn || !window.ort) { return; }
    /* absolute URL so ORT's dynamic import() of the wasm .mjs resolves (a bare
       'static/ort/...' specifier throws "Failed to resolve module specifier") */
    window.ort.env.wasm.wasmPaths = new URL('static/ort/', document.baseURI).href;
    window.ort.env.wasm.numThreads = 1;   // no cross-origin isolation needed
    Promise.all([
      fetch('static/models/region_clf.meta.json').then(function (r) { return r.json(); }),
      window.ort.InferenceSession.create('static/models/region_clf.onnx',
        { executionProviders: ['wasm'] })
    ]).then(function (res) {
      cnn.setSession(res[1], res[0]);
      cnnReady = true;
      if (detectReady()) { setStatus('', ''); }
    }).catch(function (e) {
      if (window.console) { window.console.warn('CNN model unavailable, using HOG fallback:', e); }
      if (detectReady()) { setStatus('', ''); }
    });
  }

  /**
   * Load the corner module CNN (the primary detector) + its metadata (tier
   * thresholds, corner geometry). On failure the region CNN / HOG still work.
   */
  function loadCornerModel() {
    var corners = window.DET && window.DET.corners;
    if (!corners || !window.ort) { return; }
    Promise.all([
      fetch('static/models/corner_clf.meta.json').then(function (r) { return r.json(); }),
      window.ort.InferenceSession.create('static/models/corner_clf.onnx',
        { executionProviders: ['wasm'] })
    ]).then(function (res) {
      corners.setSession(res[1], res[0]);
      cornersReady = true;
      if (detectReady()) { setStatus('', ''); }
    }).catch(function (e) {
      if (window.console) { window.console.warn('corner model unavailable, using region CNN:', e); }
      if (detectReady()) { setStatus('', ''); }
    });
  }

  /**
   * Load the YOLOX-Nano detector (ONNX) + its metadata (input size, strides,
   * conf threshold) via onnxruntime-web. This is the primary detector; on failure
   * the corner CNN / region CNN / HOG still handle detection.
   */
  function loadYoloxModel() {
    var yolox = window.DET && window.DET.yolox;
    if (!yolox || !window.ort) { return; }
    window.ort.env.wasm.wasmPaths = new URL('static/ort/', document.baseURI).href;
    window.ort.env.wasm.numThreads = 1;
    Promise.all([
      fetch('static/models/yolox.meta.json').then(function (r) { return r.json(); }),
      window.ort.InferenceSession.create('static/models/yolox_nano_rayban.onnx',
        { executionProviders: ['wasm'] })
    ]).then(function (res) {
      yolox.setSession(res[1], res[0]);
      yoloxReady = true;
      if (detectReady()) { setStatus('', ''); }
    }).catch(function (e) {
      if (window.console) { window.console.warn('YOLOX model unavailable, using corner/region pipeline:', e); }
      if (detectReady()) { setStatus('', ''); }
    });
  }

  /** Run cb once the OpenCV.js wasm runtime is initialised (race-safe). */
  function whenCvReady(cb) {
    if (window.cv && window.cv.Mat) { cb(); return; }
    if (window.cv) { window.cv.onRuntimeInitialized = cb; }
    var iv = window.setInterval(function () {
      if (window.cv && window.cv.Mat) { window.clearInterval(iv); cb(); }
    }, 50);
  }

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
   * Read the per-corner camera probabilities (0..1) the verdict is based on.
   * `has` is false when no classifier is loaded (pipeline fell back to Hough).
   * @param {Object} payload
   * @returns {{l:number, r:number, has:boolean}}
   */
  function camProbs(payload) {
    var pf = payload.per_corner || {};
    var L = pf.L || {}, R = pf.R || {};
    var ok = function (v) { return v !== null && v !== undefined; };
    return { l: L.cam_prob, r: R.cam_prob, has: ok(L.cam_prob) && ok(R.cam_prob) };
  }

  /**
   * Format a 0..1 number as a whole percentage.
   * @param {number} n
   * @returns {string}
   */
  function pct(n) {
    return Math.round(n * 100) + '%';
  }

  /**
   * Paint one corner card with its camera probability and pass/fail state.
   * @param {string} scoreId
   * @param {string} stateId
   * @param {string} cardId
   * @param {number} prob
   * @param {number} thr
   * @param {boolean} hasProb
   */
  function renderCorner(scoreId, stateId, cardId, prob, thr, hasProb, thrMaybe) {
    var sc = el(scoreId), st = el(stateId), card = el(cardId);
    if (!sc || !st || !card) { return; }
    if (!hasProb) {
      sc.textContent = 'n/a'; st.textContent = '—'; card.className = 'corner';
      return;
    }
    sc.textContent = pct(prob);
    if (prob >= thr) { st.textContent = 'CAMERA'; card.className = 'corner is-cam'; }
    else if (thrMaybe && prob >= thrMaybe) {
      st.textContent = 'camera?'; card.className = 'corner is-maybe';
    } else { st.textContent = 'no camera'; card.className = 'corner is-nocam'; }
  }

  /**
   * Build the plain-English explanation of why the verdict came out this way.
   * @param {Object} payload
   * @param {{l:number, r:number, has:boolean}} cp
   * @param {number} thr
   * @returns {string}
   */
  function whyText(payload, cp, thr) {
    var isMeta = payload.verdict === 'META';
    /* domain-gate / Hough-fallback cases: the server reason is the truth */
    if (!cp.has) { return payload.reason || ''; }
    if (!isMeta && payload.reason &&
        payload.reason.indexOf('isolated') !== -1) {
      return payload.reason;
    }
    var L = pct(cp.l), R = pct(cp.r), T = pct(thr);
    if (isMeta) {
      return 'A camera module was recognised in BOTH top-outer corners — ' +
        'left ' + L + ' and right ' + R + ' confidence (need ≥ ' + T +
        '). Ray-Ban Meta glasses carry a camera in each corner, so both ' +
        'corners passing is what flags this as smart glasses.';
    }
    if (payload.verdict === 'MAYBE') {
      var hit = (cp.l >= cp.r) ? 'left' : 'right';
      return 'One top-outer corner (' + hit + ') looks like a camera module ' +
        '(left ' + L + ', right ' + R + ') but the other does not — typical ' +
        'when the head is turned so only one corner faces the camera, or one ' +
        'side is covered. Treat as possible smart glasses and check by eye.';
    }
    var miss = (cp.l < thr && cp.r < thr) ? 'either corner' :
               (cp.l < thr ? 'the left corner' : 'the right corner');
    return 'No camera module in ' + miss + ' (left ' + L + ', right ' + R +
      '; need ≥ ' + T + '). Both corners must contain a camera for ' +
      'smart glasses, so this reads as normal eyeglasses.';
  }

  /**
   * Paint the full result panel from an API payload.
   * @param {Object} payload
   */
  function renderResult(payload) {
    var result = el('result');
    var isMeta = payload.verdict === 'META';
    var isMaybe = payload.verdict === 'MAYBE';
    var isNoGlasses = payload.verdict === 'NOGLASSES';
    var threshold = payload.threshold || 0.5;
    var cp = camProbs(payload);
    /* Corner/HOG path: the meter shows the corner the verdict hinges on — the
       weaker one for a YES call, the stronger one otherwise (that is what a
       MAYBE hangs on). Region-CNN path: the single region probability. */
    var conf = cp.has ? (isMeta ? Math.min(cp.l, cp.r) : Math.max(cp.l, cp.r))
      : (typeof payload.overall_score === 'number' ? payload.overall_score : 0);

    el('overlay').src = payload.overlay || '';
    lastOverlay = payload.overlay || null;   // kept for the DOWNLOAD button
    el('verdict-text').textContent = isMeta ? 'SMART GLASSES'
      : (isMaybe ? 'MAYBE — CHECK'
        : (isNoGlasses ? 'NO GLASSES' : 'NORMAL GLASSES'));
    el('verdict-reason').textContent = isMeta
      ? 'camera recognised in both corners'
      : (isMaybe ? 'a camera module may be present — one corner fired'
        : (isNoGlasses ? 'no eyewear detected on the face'
                       : 'not smart glasses'));

    el('conf-val').textContent = fmt(conf);
    renderCorner('score-l', 'state-l', 'corner-l', cp.l, threshold, cp.has,
      payload.threshold_maybe);
    renderCorner('score-r', 'state-r', 'corner-r', cp.r, threshold, cp.has,
      payload.threshold_maybe);
    el('why-text').textContent = whyText(payload, cp, threshold);

    /* result actions: RE-CAPTURE (camera) or SCAN ANOTHER (upload), plus DOWNLOAD */
    var again = el('again');
    if (again) {
      again.hidden = false;
      again.textContent = (mode === 'camera') ? 'RE-CAPTURE' : 'SCAN ANOTHER';
    }

    var cls = isMeta ? 'is-meta'
      : (isMaybe ? 'is-maybe' : (isNoGlasses ? 'is-noglasses' : 'is-normal'));
    result.className = 'result ' + cls;
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
      result.className = 'result ' + cls;
    }, 1100);

    setStatus('scan complete · ' + payload.segment_method + ' segmentation', '');
    result.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  /**
   * Upload an image (file picker upload OR a captured webcam frame) to the
   * detector and render the response.
   * @param {Blob|File} file       image File, or a canvas Blob (webcam frame)
   * @param {Function} [onComplete] called after the request settles (ok or error)
   */
  function scan(file, onComplete) {
    var done = function (ok) { if (typeof onComplete === 'function') { onComplete(ok); } };
    if (!file) { done(false); return; }
    if (file.type && file.type.indexOf('image/') !== 0) {
      setStatus('that is not an image file', 'error');
      done(false);
      return;
    }
    if (!detectReady()) {
      setStatus('detector still loading — one moment …', 'busy');
      done(false);
      return;
    }
    setStatus('scanning corners for a camera module', 'busy');

    var url = URL.createObjectURL(file);
    var img = new Image();
    img.onload = function () {
      /* keep the source image (re-encoded to PNG) for the DOWNLOAD button */
      try {
        var pc = document.createElement('canvas');
        pc.width = img.naturalWidth || img.width;
        pc.height = img.naturalHeight || img.height;
        pc.getContext('2d').drawImage(img, 0, 0);
        lastPhoto = pc.toDataURL('image/png');
      } catch (e) { lastPhoto = null; }
      runDetect(img).then(function (payload) {
        URL.revokeObjectURL(url);
        renderResult(payload);
        done(true);
      }, function (e) {
        URL.revokeObjectURL(url);
        setStatus('detector error: ' + (e && e.message ? e.message : e), 'error');
        done(false);
      });
    };
    img.onerror = function () {
      URL.revokeObjectURL(url);
      setStatus('could not read that image', 'error');
      done(false);
    };
    img.src = url;
  }

  /**
   * Run the full in-browser pipeline on a loaded image and return the payload
   * (same shape the old /api/detect endpoint produced), with an overlay drawn.
   * @param {HTMLImageElement} img
   * @returns {Object}
   */
  function runDetect(img) {
    var cv = window.cv;
    var w = img.naturalWidth || img.width;
    var h = img.naturalHeight || img.height;
    var c = document.createElement('canvas');
    c.width = w; c.height = h;
    var ctx = c.getContext('2d');
    /* No white fill: the Python pipeline reads images with cv2.imread(IMREAD_COLOR),
       which drops the alpha channel. A transparent canvas reads opaque pixels back
       identically to cv2 and avoids white-compositing skew on RGBA inputs. Live
       webcam frames are fully opaque, so this path is exact for camera mode. */
    ctx.drawImage(img, 0, 0, w, h);
    var id = ctx.getImageData(0, 0, w, h);
    var src = cv.matFromImageData(id);
    /* Primary: YOLOX-Nano on the FULL-RES mat (box + confidence -> META/NORMAL),
       the same model run server-side. It always resolves to a payload. Fallbacks
       (only if YOLOX failed to load): corner CNN -> region CNN -> legacy HOG. */
    var pending;
    if (yoloxReady) {
      pending = window.DET.yolox.detect(src, cv);
    } else if (cornersReady) {
      pending = window.DET.corners.detect(src, cv).then(function (payload) {
        if (payload) { return payload; }
        if (cnnReady) { return window.DET.detectRegion(src, cv, window.DET.config); }
        if (model) { return window.DET.detectMat(src, cv, model, window.DET.config); }
        throw new Error('no face found and no fallback detector loaded');
      });
    } else if (cnnReady) {
      pending = window.DET.detectRegion(src, cv, window.DET.config);
    } else {
      pending = Promise.resolve(window.DET.detectMat(src, cv, model, window.DET.config));
    }
    return pending.then(function (payload) {
      src.delete();
      payload.overlay = buildOverlay(img, payload);
      return payload;
    }, function (err) {
      src.delete();
      throw err;
    });
  }

  /**
   * Draw the annotated overlay (frame bbox, lens boxes, corner ROIs + P(camera),
   * verdict banner) on a canvas at canonical scale and return a data URI. This
   * reproduces pipeline/viz.annotate in the browser.
   * @param {HTMLImageElement} img
   * @param {Object} payload
   * @returns {string} PNG data URI
   */
  function buildOverlay(img, payload) {
    var g = payload.geom || {};
    var W = g.colorW || 1000, H = g.colorH || 1000;
    var c = document.createElement('canvas');
    c.width = W; c.height = H;
    var ctx = c.getContext('2d');
    ctx.drawImage(img, 0, 0, W, H);

    function rect(x, y, w, h, color, lw) {
      ctx.strokeStyle = color; ctx.lineWidth = lw || 2;
      ctx.strokeRect(x, y, w, h);
    }
    /* CNN path: draw the located glasses region the model classified */
    if (g.region) {
      var isM = payload.verdict === 'META';
      rect(g.region[0], g.region[1], g.region[2], g.region[3], isM ? '#ff4d4d' : '#c6f24e', 3);
    }
    var bb = g.bbox || payload.bbox;
    if (bb && !g.region) { rect(bb[0], bb[1], bb[2], bb[3], '#2d7dff', 2); }
    /* YOLOX path: mark each camera module at the top-outer end-pieces */
    (g.cameras || []).forEach(function (pt) {
      var rC = Math.max(6, Math.round(0.03 * ((g.region && g.region[2]) || 40)));
      ctx.strokeStyle = '#ff4d4d'; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(pt[0], pt[1], rC, 0, Math.PI * 2); ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(pt[0] - rC, pt[1]); ctx.lineTo(pt[0] + rC, pt[1]);
      ctx.moveTo(pt[0], pt[1] - rC); ctx.lineTo(pt[0], pt[1] + rC); ctx.stroke();
      ctx.fillStyle = '#ff4d4d'; ctx.font = '14px "Space Mono", monospace';
      ctx.fillText('camera', pt[0] - rC, Math.max(12, pt[1] - rC - 4));
    });
    [g.lensL, g.lensR].forEach(function (ln) {
      if (!ln) { return; }
      rect(ln.cx - ln.hw, ln.cy - ln.hh, ln.hw * 2, ln.hh * 2, '#27e08a', 1);
    });
    (g.rois || []).forEach(function (roi) {
      rect(roi.x, roi.y, roi.w, roi.h, '#ffd23f', 2);
      var p = (payload.per_corner[roi.side] || {}).cam_prob;
      ctx.fillStyle = '#ffd23f';
      ctx.font = '14px "Space Mono", monospace';
      ctx.fillText(roi.side + (p == null ? '' : '  Pcam=' + fmt(p)),
        roi.x, Math.max(12, roi.y - 4));
    });
    /* face box (corner-detector path) */
    if (g.face) { rect(g.face[0], g.face[1], g.face[2], g.face[3], '#2d7dff', 1); }
    /* header banner */
    var isMeta = payload.verdict === 'META';
    var isMaybe = payload.verdict === 'MAYBE';
    var isNoG = payload.verdict === 'NOGLASSES';
    ctx.fillStyle = 'rgba(0,0,0,0.78)';
    ctx.fillRect(0, 0, W, 28);
    ctx.fillStyle = isMeta ? '#ff4d4d'
      : (isMaybe ? '#ffb03f' : (isNoG ? '#9fb4c7' : '#27e08a'));
    ctx.font = '700 16px "Space Mono", monospace';
    var pL = payload.prob_left, pR = payload.prob_right;
    var banner;
    if (g.region) {
      banner = payload.verdict + '  P=' + fmt(payload.overall_score || 0);
    } else {
      banner = payload.verdict + (pL == null ? '' :
        '  Pcam L=' + fmt(pL) + ' R=' + fmt(pR));
    }
    ctx.fillText(banner, 6, 20);
    return c.toDataURL('image/png');
  }

  /* ======================================================================
     LIVE CAMERA MODE — single-shot capture
     enter → live preview + CAPTURE PHOTO → grab one frame → analyse → verdict.
     The result panel offers RE-CAPTURE (back to the live feed) and DOWNLOAD
     (the plain captured photo + the annotated result image).
     ====================================================================== */

  var stream = null;      // the active MediaStream (kept across captures)
  var timers = [];        // pending setTimeout handles (cleared on mode switch)
  /* the most recent capture + its annotated overlay, for the DOWNLOAD button */
  var lastPhoto = null;   // PNG data URI of the captured/uploaded frame
  var lastOverlay = null; // PNG data URI of the annotated result

  /** Clear every pending timer so a mode switch fully stops any deferred work. */
  function clearTimers() {
    var i;
    for (i = 0; i < timers.length; i += 1) { window.clearTimeout(timers[i]); }
    timers = [];
  }

  /** Toggle the [hidden] attribute on an element by id (null-safe). */
  function show(id, visible) {
    var n = el(id);
    if (n) { n.hidden = !visible; }
  }

  /**
   * Enter live-camera mode: request the webcam, then show the live preview with
   * the CAPTURE PHOTO control straight away. No audio, no clip, no recording.
   */
  function enterCamera() {
    mode = 'camera';
    clearTimers();
    show('camera-mode', true);
    show('upload-mode', false);
    show('live-wrap', false);
    show('cam-controls', false);
    show('cam-instruction', true);
    el('cam-instruction').textContent =
      'Initialising camera … allow access when prompted.';

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setStatus('this browser has no camera access; use Image Upload', 'error');
      return;
    }
    navigator.mediaDevices.getUserMedia({ video: true })
      .then(function (s) {
        if (mode !== 'camera') { s.getTracks().forEach(function (t) { t.stop(); }); return; }
        stream = s;
        readyToScan();
      })
      .catch(function () {
        setStatus('camera access denied — switch to Image Upload instead', 'error');
      });
  }

  /**
   * Show the live self-view preview plus the CAPTURE PHOTO control, ready for
   * the user to take a shot whenever they like.
   */
  function readyToScan() {
    if (mode !== 'camera' || !stream) { return; }
    show('cam-instruction', false);
    var live = el('live');
    live.srcObject = stream;
    el('cam-state').textContent = '';
    show('live-wrap', true);
    show('cam-controls', true);
    setStatus('press CAPTURE PHOTO to scan', '');
  }

  /** Leave camera mode: stop any deferred work, release the webcam, reset UI. */
  function leaveCamera() {
    clearTimers();
    if (stream) {
      stream.getTracks().forEach(function (t) { t.stop(); });
      stream = null;
    }
    var live = el('live');
    if (live) { live.srcObject = null; }
    show('camera-mode', false);
  }

  /** Trigger a download of a data URI / blob URL under the given filename. */
  function triggerDownload(href, filename) {
    var a = document.createElement('a');
    a.href = href;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  }

  /**
   * CAPTURE PHOTO handler: grab the current live frame, keep it for download,
   * then run the same detector/render path uploads use. One frame, no countdown.
   */
  function captureAndAnalyze() {
    if (mode !== 'camera' || !stream) { return; }
    var live = el('live');
    var canvas = grabLiveFrame(live);
    lastPhoto = canvas.toDataURL('image/png');
    show('cam-controls', false);
    el('cam-state').textContent = 'captured · analysing …';
    setStatus('scanning corners for a camera module', 'busy');
    runDetect(canvas).then(function (payload) {
      if (mode !== 'camera') { return; }
      /* hide the live panel so the result stands alone (webcam keeps streaming;
         RE-CAPTURE brings the feed back) */
      show('camera-mode', false);
      renderResult(payload);
    }, function (e) {
      if (mode !== 'camera') { return; }
      setStatus('detector error: ' + (e && e.message ? e.message : e), 'error');
      show('cam-controls', true);
    });
  }

  /** RE-CAPTURE: drop the last verdict and return to the live preview. */
  function recapture() {
    var result = el('result');
    if (result) { result.hidden = true; }
    document.body.className = '';
    el('cam-state').textContent = '';
    show('camera-mode', true);
    show('live-wrap', true);
    show('cam-controls', true);
    setStatus('press CAPTURE PHOTO to scan', '');
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  /**
   * Grab the current live video frame to a fresh canvas, mirrored to match the
   * selfie-mirrored preview (CSS scaleX(-1) on #live) so a saved frame matches
   * what the user saw. Detection is unaffected — the classifier is flip-invariant.
   * @param {HTMLVideoElement} live
   * @returns {HTMLCanvasElement}
   */
  function grabLiveFrame(live) {
    var w = live.videoWidth || 1280;
    var h = live.videoHeight || 720;
    var canvas = document.createElement('canvas');
    canvas.width = w;
    canvas.height = h;
    var g = canvas.getContext('2d');
    g.save();
    g.translate(w, 0);
    g.scale(-1, 1);
    g.drawImage(live, 0, 0, w, h);
    g.restore();
    return canvas;
  }

  /** Switch input mode and (de)activate the camera accordingly. */
  function setMode(next) {
    if (next === mode) { return; }
    var camTab = el('tab-camera');
    var upTab = el('tab-upload');
    if (next === 'upload') {
      leaveCamera();             // leaveCamera reads `mode`, so flip it after
      mode = 'upload';
      show('upload-mode', true);
      if (camTab) { camTab.className = 'mode-tab'; camTab.setAttribute('aria-selected', 'false'); }
      if (upTab) { upTab.className = 'mode-tab is-active'; upTab.setAttribute('aria-selected', 'true'); }
      var result = el('result');
      if (result) { result.hidden = true; }
      document.body.className = '';
      setStatus('', '');
    } else {
      if (upTab) { upTab.className = 'mode-tab'; upTab.setAttribute('aria-selected', 'false'); }
      if (camTab) { camTab.className = 'mode-tab is-active'; camTab.setAttribute('aria-selected', 'true'); }
      enterCamera();
    }
  }

  /**
   * Wire up the drop zone, file input and reset button.
   */
  function init() {
    var drop = el('drop');
    var input = el('file');
    var again = el('again');

    /* load models + the OpenCV.js runtime up front. Primary: the region CNN
       (onnxruntime-web). Fallback: the legacy HOG corner classifier JSON. */
    setStatus('loading detector …', 'busy');
    fetch('static/camera_clf.json')
      .then(function (r) { return r.json(); })
      .then(function (m) { model = m; if (detectReady()) { setStatus('', ''); } })
      .catch(function () { /* HOG is only a fallback; CNN may still load */ });
    loadYoloxModel();   // YOLOX-Nano — the primary detector (async, own readiness)
    loadCnnModel();     // region CNN fallback (async, own readiness)
    loadCornerModel();  // corner module CNN — fallback if YOLOX fails to load
    whenCvReady(function () {
      cvReady = true;
      loadFaceCascades(window.cv);   // face + eye cascades; needs the wasm runtime up
      if (detectReady()) { setStatus('', ''); }
    });

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

    /* RE-CAPTURE (camera) returns to the live feed; SCAN ANOTHER (upload) resets */
    if (again) {
      again.addEventListener('click', function () {
        if (mode === 'camera') { recapture(); return; }
        var result = el('result');
        if (result) { result.hidden = true; }
        document.body.className = '';
        if (input) { input.value = ''; }
        setStatus('', '');
        window.scrollTo({ top: 0, behavior: 'smooth' });
      });
    }

    /* DOWNLOAD saves both the plain captured/uploaded photo and the annotated
       result image, staggered so the browser doesn't collapse them into one */
    var download = el('download');
    if (download) {
      download.addEventListener('click', function () {
        var n = 0;
        if (lastPhoto) { triggerDownload(lastPhoto, 'capture-photo.png'); n += 1; }
        if (lastOverlay) {
          window.setTimeout(function () {
            triggerDownload(lastOverlay, 'capture-result.png');
          }, 350);
          n += 1;
        }
        setStatus(n ? 'downloading ' + n + ' file' + (n === 1 ? '' : 's') + ' …'
                    : 'nothing to download yet', n ? '' : 'error');
      });
    }

    /* mode tabs */
    var tabCam = el('tab-camera');
    var tabUp = el('tab-upload');
    if (tabCam) { tabCam.addEventListener('click', function () { setMode('camera'); }); }
    if (tabUp) { tabUp.addEventListener('click', function () { setMode('upload'); }); }

    /* CAPTURE PHOTO grabs one frame and analyses it */
    var camStart = el('cam-start');
    if (camStart) { camStart.addEventListener('click', captureAndAnalyze); }

    /* camera is the default mode on load */
    enterCamera();
  }

  document.addEventListener('DOMContentLoaded', init);
})();
