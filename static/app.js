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
    var threshold = payload.threshold || 0.5;
    var cp = camProbs(payload);
    /* Corner/HOG path: the meter shows the corner the verdict hinges on — the
       weaker one for a YES call, the stronger one otherwise (that is what a
       MAYBE hangs on). Region-CNN path: the single region probability. */
    var conf = cp.has ? (isMeta ? Math.min(cp.l, cp.r) : Math.max(cp.l, cp.r))
      : (typeof payload.overall_score === 'number' ? payload.overall_score : 0);

    el('overlay').src = payload.overlay || '';
    el('verdict-text').textContent = isMeta ? 'SMART GLASSES'
      : (isMaybe ? 'MAYBE — CHECK' : 'NORMAL GLASSES');
    el('verdict-reason').textContent = isMeta
      ? 'camera recognised in both corners'
      : (isMaybe ? 'a camera module may be present — one corner fired'
                 : 'not smart glasses');

    el('conf-val').textContent = fmt(conf);
    renderCorner('score-l', 'state-l', 'corner-l', cp.l, threshold, cp.has,
      payload.threshold_maybe);
    renderCorner('score-r', 'state-r', 'corner-r', cp.r, threshold, cp.has,
      payload.threshold_maybe);
    el('why-text').textContent = whyText(payload, cp, threshold);

    /* the loop reruns automatically in camera mode, so hide the manual reset */
    var again = el('again');
    if (again) { again.hidden = (mode === 'camera'); }

    var cls = isMeta ? 'is-meta' : (isMaybe ? 'is-maybe' : 'is-normal');
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
    ctx.fillStyle = 'rgba(0,0,0,0.78)';
    ctx.fillRect(0, 0, W, 28);
    ctx.fillStyle = isMeta ? '#ff4d4d' : (isMaybe ? '#ffb03f' : '#27e08a');
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
     LIVE CAMERA MODE
     The how-it-works clip plays in the intro column (a separate place), while
     the camera scene shows the live feed straight away:
     enter → live preview + START (clip auto-plays alongside)
           → START → [5s countdown → grab → analyse]∞
     ====================================================================== */

  var CYCLE_SECONDS = 5;       // live-preview countdown before each capture
  var RESULT_HOLD_MS = 3000;   // how long a verdict stays before the next cycle
  var BURST_FRAMES = 5;        // frames grabbed per analysed pose (CNN path)
  var BURST_GAP_MS = 200;      // spacing between burst frames (~1s total window)

  /* The guided capture sequence: one entry per pose. Pressing START walks these
     in order — show this pose, count down 5s, grab a frame — then stops by
     itself after the last pose. Only poses with analyze:true run the detector;
     the left/right side frames are just captured and kept (see `captures`) for
     later use. `label`/`arrow` drive the on-screen "which side" banner. */
  var STEPS = [
    { key: 'left',  analyze: false, arrow: '◀', side: 'left',
      label: 'LEFT SIDE',
      prompt: 'Turn the glasses so their LEFT side faces the camera' },
    { key: 'right', analyze: false, arrow: '▶', side: 'right',
      label: 'RIGHT SIDE',
      prompt: 'Now turn so the RIGHT side faces the camera' },
    { key: 'straight', analyze: true, arrow: '', side: '',
      label: 'FRONT · STRAIGHT ON',
      prompt: 'Now hold the glasses straight on, facing the camera' }
  ];

  /* the front/straight pose on its own — used for "CAPTURE AGAIN", which re-runs
     only the analysed pose and skips the left/right side captures */
  var STRAIGHT_STEP = STEPS[STEPS.length - 1];

  var stream = null;           // the active MediaStream (kept across cycles)
  var screenStream = null;     // the shared-tab MediaStream being recorded
  var recorder = null;         // MediaRecorder over screenStream
  var recordedChunks = [];     // accumulated recording data blobs
  var recordingActive = false; // true while the screen recording is running
  var timers = [];             // pending setTimeout handles
  var ticker = null;           // the countdown setInterval handle
  var sequence = STEPS;        // the pose list the current run walks
  var stepIndex = 0;           // which pose in `sequence` is being captured
  var captures = [];           // {key, dataUrl} per captured pose, kept for later
  var didFullRun = false;      // true once the full left/right/straight run is done
  var spokenOnce = false;      // the instructions narration plays only once per load

  /* expose the captured pose frames so later steps can pick them up */
  window.SCAN_CAPTURES = captures;

  /**
   * Speak the instructions once. Browsers block autoplay-with-sound until a
   * user gesture, so if play() is rejected we retry on the first tap/click.
   */
  function playInstructions() {
    if (spokenOnce) { return; }
    var audio = el('intro-audio');
    if (!audio) { return; }
    spokenOnce = true;
    var fired = false;
    var go = function () {
      if (fired) { return; }
      fired = true;
      try { audio.currentTime = 0; } catch (e) {}
      audio.play();
    };
    var p = audio.play();
    if (p && typeof p.catch === 'function') {
      p.catch(function () {
        /* autoplay blocked — arm a one-shot gesture listener to start it */
        var unlock = function () {
          document.removeEventListener('pointerdown', unlock);
          document.removeEventListener('keydown', unlock);
          go();
        };
        document.addEventListener('pointerdown', unlock);
        document.addEventListener('keydown', unlock);
      });
    }
  }

  /** Clear every pending timer/interval so a mode switch fully stops the loop. */
  function clearTimers() {
    var i;
    for (i = 0; i < timers.length; i += 1) { window.clearTimeout(timers[i]); }
    timers = [];
    if (ticker !== null) { window.clearInterval(ticker); ticker = null; }
  }

  /** setTimeout that registers its handle for clearTimers(). */
  function later(fn, ms) {
    var id = window.setTimeout(fn, ms);
    timers.push(id);
    return id;
  }

  /** Toggle the [hidden] attribute on an element by id (null-safe). */
  function show(id, visible) {
    var n = el(id);
    if (n) { n.hidden = !visible; }
  }

  /**
   * Enter live-camera mode: request the webcam, then show the live preview with
   * START straight away in the camera scene while the how-it-works clip plays in
   * the intro column. The clip never occupies the camera scene.
   */
  function enterCamera() {
    mode = 'camera';
    clearTimers();
    /* fresh entry → next run is the full left/right/straight sequence */
    didFullRun = false;
    setStartLabel('START&nbsp;TEST');
    show('cam-end', false);
    show('camera-mode', true);
    show('upload-mode', false);
    show('live-wrap', false);
    show('cam-controls', false);
    show('cam-instruction', true);
    el('cam-instruction').textContent =
      'Initialising camera … allow access when prompted.';

    playIntro();          // clip lives in the intro column, independent of the feed
    playInstructions();   // spoken instructions, once per page load

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setStatus('this browser has no camera access; use Upload Image', 'error');
      return;
    }
    navigator.mediaDevices.getUserMedia({ video: true })
      .then(function (s) {
        if (mode !== 'camera') { s.getTracks().forEach(function (t) { t.stop(); }); return; }
        stream = s;
        readyToScan();
      })
      .catch(function () {
        setStatus('camera access denied — switch to Upload Image instead', 'error');
      });
  }

  /**
   * Show the live self-view preview plus the START control, ready for the user
   * to begin scanning whenever they like.
   */
  function readyToScan() {
    if (mode !== 'camera' || !stream) { return; }
    show('cam-instruction', false);
    var live = el('live');
    live.srcObject = stream;
    el('countdown').hidden = true;
    el('cam-state').textContent = '';
    show('live-wrap', true);
    show('cam-controls', true);
    setStatus('watch the how-it-works clip, then press START TEST to scan', '');
  }

  /**
   * Play the instructional clip from the start, in its own panel in the intro
   * column. Used on camera entry; failures (blocked autoplay) are silent.
   */
  function playIntro() {
    show('intro-clip', true);
    var intro = el('intro');
    if (!intro) { return; }
    intro.currentTime = 0;
    var p = intro.play();
    if (p && typeof p.catch === 'function') { p.catch(function () {}); }
  }

  /** Leave camera mode: stop the loop, release the webcam, reset the UI. */
  function leaveCamera() {
    clearTimers();
    var intro = el('intro');
    if (intro) { intro.pause(); }
    show('intro-clip', false);
    if (stream) {
      stream.getTracks().forEach(function (t) { t.stop(); });
      stream = null;
    }
    /* release the shared tab if a recording is still running (discarded) */
    if (recorder && recorder.state !== 'inactive') {
      recorder.onstop = null;
      try { recorder.stop(); } catch (e) {}
    }
    if (screenStream) {
      screenStream.getTracks().forEach(function (t) { t.stop(); });
      screenStream = null;
    }
    recorder = null;
    recordingActive = false;
    show('cam-end', false);
    var live = el('live');
    if (live) { live.srcObject = null; }
    show('camera-mode', false);
  }

  /**
   * Ask the user to share this tab and start recording it. Must be called from a
   * user gesture (browsers block screen-share otherwise). Resolves to true once
   * recording is running, false if unsupported or the user cancels the prompt.
   * @returns {Promise<boolean>}
   */
  function startScreenRecording() {
    if (recordingActive) { return Promise.resolve(true); }
    if (!navigator.mediaDevices || !navigator.mediaDevices.getDisplayMedia ||
        typeof window.MediaRecorder === 'undefined') {
      return Promise.resolve(false);
    }
    /* preferCurrentTab nudges Chrome to offer THIS tab first */
    var opts = { video: { frameRate: 30 }, audio: false, preferCurrentTab: true };
    return navigator.mediaDevices.getDisplayMedia(opts).then(function (s) {
      screenStream = s;
      recordedChunks = [];
      var mime = '';
      var prefs = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm'];
      for (var i = 0; i < prefs.length; i += 1) {
        if (window.MediaRecorder.isTypeSupported &&
            window.MediaRecorder.isTypeSupported(prefs[i])) { mime = prefs[i]; break; }
      }
      recorder = mime ? new window.MediaRecorder(s, { mimeType: mime })
                      : new window.MediaRecorder(s);
      recorder.ondataavailable = function (e) {
        if (e.data && e.data.size > 0) { recordedChunks.push(e.data); }
      };
      recorder.start();
      recordingActive = true;
      /* if the user stops sharing from the browser's own UI, reflect that */
      s.getVideoTracks().forEach(function (t) {
        t.addEventListener('ended', function () { recordingActive = false; });
      });
      return true;
    }).catch(function () { return false; });
  }

  /**
   * Stop the screen recording, assemble the captured chunks into a WebM blob and
   * trigger a download. Safe to call when nothing was recorded.
   */
  function stopRecordingAndDownload() {
    var finalize = function () {
      var type = (recordedChunks[0] && recordedChunks[0].type) || 'video/webm';
      var blob = new Blob(recordedChunks, { type: type });
      if (screenStream) {
        screenStream.getTracks().forEach(function (t) { t.stop(); });
        screenStream = null;
      }
      recordingActive = false;
      recorder = null;
      if (!blob.size) {
        setStatus('no screen recording was captured', 'error');
        return;
      }
      var url = URL.createObjectURL(blob);
      var a = document.createElement('a');
      a.href = url;
      a.download = 'module-scan-recording.webm';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.setTimeout(function () { URL.revokeObjectURL(url); }, 4000);
      setStatus('recording downloaded · module-scan-recording.webm', '');
    };
    if (recorder && recorder.state !== 'inactive') {
      recorder.onstop = finalize;
      recorder.stop();
    } else {
      finalize();
    }
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
   * Download every captured pose photo. Each is a PNG data URI in `captures`,
   * named by capture order and pose key (e.g. 01-left.png, 03-straight.png).
   * Downloads are staggered so browsers don't collapse them into one.
   * @returns {number} how many photos were queued for download
   */
  function downloadCaptures() {
    captures.forEach(function (cap, i) {
      var n = (i + 1 < 10 ? '0' : '') + (i + 1);
      window.setTimeout(function () {
        triggerDownload(cap.dataUrl, n + '-' + cap.key + '.png');
      }, i * 350);
    });
    return captures.length;
  }

  /** START handler: begin the guided three-pose capture sequence. */
  function startLoop() {
    if (mode !== 'camera' || !stream) { return; }
    show('cam-controls', false);
    show('cam-instruction', false);
    var live = el('live');
    live.srcObject = stream;
    show('cam-end', false);

    var begin = function () {
      if (mode !== 'camera') { return; }
      show('cam-controls', false);
      /* first run walks all three poses; after that, "CAPTURE AGAIN" re-runs
         only the straight pose through analysis */
      sequence = didFullRun ? [STRAIGHT_STEP] : STEPS;
      stepIndex = 0;
      /* full run starts a fresh capture set; recaptures append, so every photo
         taken this session (left, right, straight + any retakes) is kept and
         downloaded on END */
      if (!didFullRun) { captures.length = 0; }
      runCycle();
    };

    /* recording runs continuously across the whole session; the share-tab
       prompt only appears the first time (browsers require this click) */
    if (recordingActive) { begin(); return; }
    setStatus('share this tab when prompted, to record the session …', 'busy');
    startScreenRecording().then(function (ok) {
      if (mode !== 'camera') { return; }
      if (!ok) { setStatus('continuing without screen recording', ''); }
      begin();
    });
  }

  /**
   * Paint the big "which side" banner over the live feed for the current pose
   * (a directional arrow on left/right poses, just a label for the front pose).
   * @param {Object} step  one entry from STEPS
   */
  function renderPoseBanner(step) {
    var banner = el('pose-banner');
    if (!banner) { return; }
    var label = '<span class="pose-banner__label">' + step.label + '</span>';
    var arrow = step.arrow
      ? '<span class="pose-banner__arrow pose-banner__arrow--' + step.side +
        '" aria-hidden="true">' + step.arrow + '</span>'
      : '';
    /* arrow leads on the left pose, trails on the right pose */
    banner.innerHTML = (step.side === 'right') ? (label + arrow) : (arrow + label);
    banner.hidden = false;
  }

  /**
   * One step of the guided sequence: show the current pose prompt, run a 5s
   * countdown over the live feed, then grab + analyse a frame. Falls through to
   * finishSequence() once every pose in STEPS has been captured.
   */
  function runCycle() {
    if (mode !== 'camera') { return; }
    var step = sequence[stepIndex];
    if (!step) { finishSequence(); return; }

    /* clear any prior verdict and bring the live preview back */
    var result = el('result');
    if (result) { result.hidden = true; }
    document.body.className = '';
    show('live-wrap', true);

    renderPoseBanner(step);
    el('cam-state').textContent = 'Pose ' + (stepIndex + 1) + ' of ' +
      sequence.length + ' · ' + step.prompt;

    var remaining = CYCLE_SECONDS;
    var cd = el('countdown');
    cd.textContent = String(remaining);
    cd.hidden = false;
    setStatus(step.prompt + ' — capturing in ' + remaining + 's …', '');

    ticker = window.setInterval(function () {
      remaining -= 1;
      if (remaining > 0) {
        cd.textContent = String(remaining);
        setStatus(step.prompt + ' — capturing in ' + remaining + 's …', '');
        return;
      }
      window.clearInterval(ticker);
      ticker = null;
      cd.hidden = true;
      captureFrame();
    }, 1000);
  }

  /**
   * End the guided sequence: stop the loop, leave the final verdict on screen
   * and bring back the START controls so the user can run the three poses again.
   */
  function finishSequence() {
    if (mode !== 'camera') { return; }
    clearTimers();
    el('countdown').hidden = true;
    el('pose-banner').hidden = true;
    var poses = sequence.length;
    /* the first full run unlocks "CAPTURE AGAIN", which re-captures and
       re-analyses only the straight pose */
    didFullRun = true;
    setStartLabel('CAPTURE&nbsp;AGAIN');
    el('cam-state').textContent = poses > 1
      ? 'Sequence complete · captured ' + poses + ' poses'
      : 'Front photo captured';
    show('cam-controls', true);
    show('cam-end', recordingActive);
    setStatus('press CAPTURE AGAIN to retake the straight photo, or END to ' +
      'download the recording', '');
  }

  /**
   * Set the START / CAPTURE AGAIN button's label. Both states are green; only
   * END is red, so the label is all that changes here.
   * @param {string} text  button label (may contain entities)
   */
  function setStartLabel(text) {
    var btn = el('cam-start');
    if (btn) { btn.innerHTML = text; }
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

  /**
   * Grab a burst of `n` live frames spaced `gapMs` apart (the video keeps playing,
   * so they are genuinely different frames over a ~1s window). Uses later() so a
   * mode switch / clearTimers() aborts cleanly. Resolves to the canvas array.
   * @returns {Promise<HTMLCanvasElement[]>}
   */
  function captureBurst(live, n, gapMs) {
    return new Promise(function (resolve) {
      var frames = [];
      var grabNext = function () {
        if (mode !== 'camera') { resolve(frames); return; }
        frames.push(grabLiveFrame(live));
        el('cam-state').textContent = 'captured · analysing frame ' +
          frames.length + '/' + n + ' …';
        if (frames.length >= n) { resolve(frames); return; }
        later(grabNext, gapMs);
      };
      grabNext();
    });
  }

  /**
   * Run the CNN detector on each burst frame SEQUENTIALLY (ORT wasm is
   * single-threaded; sequential runs bound cv.Mat memory). Frames that error are
   * skipped. Resolves to [{payload, prob}] for the frames that succeeded.
   * @returns {Promise<Array<{payload:Object, prob:number}>>}
   */
  function detectBurst(frames) {
    var results = [];
    var chain = Promise.resolve();
    frames.forEach(function (canvas) {
      chain = chain.then(function () {
        if (mode !== 'camera') { return; }
        return runDetect(canvas).then(function (payload) {
          results.push({ payload: payload,
            prob: (typeof payload.overall_score === 'number' ? payload.overall_score : 0) });
        }, function () { /* skip a failed frame */ });
      });
    });
    return chain.then(function () { return results; });
  }

  /** Median of a numeric array (average of the middle two when even). */
  function median(arr) {
    var s = arr.slice().sort(function (a, b) { return a - b; });
    var mid = Math.floor(s.length / 2);
    return (s.length % 2) ? s[mid] : (s[mid - 1] + s[mid]) / 2;
  }

  /**
   * Aggregate burst results by MEDIAN probability. For an odd count the median is
   * exactly the majority vote (median >= thr iff most frames >= thr) but also
   * yields a calibrated meter value and shrugs off <=2 outlier frames (blur,
   * blink, face-detect wobble). Corner-detector frames aggregate PER SIDE and the
   * three-tier verdict is recomputed on the median pair; region-CNN frames
   * aggregate the single region probability. Returns the representative frame's
   * payload with its verdict/score/reason overwritten, or null if none.
   */
  function aggregateBurst(results) {
    if (!results.length) { return null; }
    var corner = results.filter(function (r) {
      return r.payload.per_corner && r.payload.per_corner.L &&
        typeof r.payload.per_corner.L.cam_prob === 'number';
    });
    /* corner path on a majority of frames: median per side + tier verdict */
    if (corner.length >= results.length / 2 && window.DET.corners &&
        window.DET.corners.ready()) {
      var mL = median(corner.map(function (r) { return r.payload.per_corner.L.cam_prob; }));
      var mR = median(corner.map(function (r) { return r.payload.per_corner.R.cam_prob; }));
      var v = window.DET.corners.tier(mL, mR);
      var repC = corner[Math.floor(corner.length / 2)].payload;
      var r3 = function (p) { return Math.round(p * 1000) / 1000; };
      repC.per_corner = { L: { cam_prob: r3(mL) }, R: { cam_prob: r3(mR) } };
      repC.prob_left = r3(mL); repC.prob_right = r3(mR);
      repC.overall_score = r3(Math.max(mL, mR));
      repC.verdict = (v === 'YES') ? 'META' : (v === 'MAYBE' ? 'MAYBE' : 'NORMAL');
      repC.tier = v;
      repC.reason = {
        YES: 'camera module recognised in BOTH corners',
        MAYBE: 'a camera module may be present (one corner) — check manually',
        NO: 'no camera module found'
      }[v] + ' — median of ' + corner.length + '/' + BURST_FRAMES +
        ' frames (L=' + r3(mL) + ' R=' + r3(mR) + ')';
      return repC;
    }
    /* region-CNN path: median of the single probability */
    var sorted = results.slice().sort(function (a, b) { return a.prob - b.prob; });
    var mid = Math.floor(sorted.length / 2);
    var medProb = (sorted.length % 2)
      ? sorted[mid].prob
      : (sorted[mid - 1].prob + sorted[mid].prob) / 2;
    var rep = sorted[mid].payload;              // a real frame near the median (for overlay)
    var thr = (typeof rep.threshold === 'number') ? rep.threshold : 0.5;
    var isMeta = medProb >= thr;
    var pctP = Math.round(medProb * 100) / 100;
    rep.overall_score = medProb;
    rep.verdict = isMeta ? 'META' : 'NORMAL';
    rep.reason = (isMeta ? 'camera glasses recognised' : 'no camera module found') +
      ' — median P=' + pctP + ' over ' + results.length + '/' + BURST_FRAMES + ' frames';
    return rep;
  }

  /**
   * Grab the current live frame(s), keep them in `captures`, and — only for poses
   * flagged analyze:true — run the detector. Left/right side frames are captured
   * but NOT analysed (kept for later use). The front pose gets a verdict: with the
   * CNN it is a 5-frame burst reduced to the median probability (kills flickery
   * single-frame false positives); the legacy HOG path stays single-frame.
   */
  function captureFrame() {
    if (mode !== 'camera') { return; }
    var step = sequence[stepIndex];
    var live = el('live');
    el('pose-banner').hidden = true;

    /* advance to the next pose after `holdMs`, or stop after the last one */
    var advance = function (holdMs) {
      if (mode !== 'camera') { return; }
      stepIndex += 1;
      later(stepIndex < sequence.length ? runCycle : finishSequence, holdMs);
    };

    if (!step.analyze) {
      /* left / right side: capture one frame, no detection */
      captures.push({ key: step.key, dataUrl: grabLiveFrame(live).toDataURL('image/png') });
      el('cam-state').textContent = step.label + ' captured ✓';
      setStatus(step.label + ' captured — hold on …', '');
      advance(RESULT_HOLD_MS);
      return;
    }

    /* front pose. Corner/region CNN available -> 5-frame burst + median vote. */
    if (cornersReady || cnnReady) {
      el('cam-state').textContent = 'captured · analysing …';
      setStatus('analysing a burst of frames …', 'busy');
      captureBurst(live, BURST_FRAMES, BURST_GAP_MS).then(function (frames) {
        if (mode !== 'camera' || !frames.length) { return advance(RESULT_HOLD_MS); }
        /* keep every burst frame for later use / hard-example collection */
        frames.forEach(function (c, i) {
          captures.push({ key: step.key + '-b' + (i + 1),
            dataUrl: c.toDataURL('image/png') });
        });
        detectBurst(frames).then(function (results) {
          if (mode !== 'camera') { return; }
          var payload = aggregateBurst(results);
          if (!payload) {
            setStatus('all burst frames failed to analyse — retrying next cycle', 'error');
            advance(RESULT_HOLD_MS);
            return;
          }
          renderResult(payload);
          advance(RESULT_HOLD_MS);
        });
      });
      return;
    }

    /* legacy HOG fallback: single-frame verdict (payloads carry no overall_score) */
    var canvas = grabLiveFrame(live);
    captures.push({ key: step.key, dataUrl: canvas.toDataURL('image/png') });
    el('cam-state').textContent = 'captured · analysing …';
    if (canvas.toBlob) {
      canvas.toBlob(function (blob) {
        if (mode !== 'camera') { return; }
        scan(blob, function () { advance(RESULT_HOLD_MS); });
      }, 'image/png');
    } else {
      advance(RESULT_HOLD_MS);  // ancient browser without toBlob
    }
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

    /* mode tabs */
    var tabCam = el('tab-camera');
    var tabUp = el('tab-upload');
    if (tabCam) { tabCam.addEventListener('click', function () { setMode('camera'); }); }
    if (tabUp) { tabUp.addEventListener('click', function () { setMode('upload'); }); }

    /* START TEST button kicks off the live capture loop */
    var camStart = el('cam-start');
    if (camStart) { camStart.addEventListener('click', startLoop); }

    /* END stops the screen recording and downloads it, then resets for a
       fresh session */
    var camEnd = el('cam-end');
    if (camEnd) {
      camEnd.addEventListener('click', function () {
        clearTimers();
        var nPhotos = downloadCaptures();   // all captured pose photos
        stopRecordingAndDownload();         // plus the screen recording
        setStatus('downloading ' + nPhotos + ' photo' +
          (nPhotos === 1 ? '' : 's') + ' + the screen recording …', '');
        didFullRun = false;
        setStartLabel('START&nbsp;TEST');
        show('cam-end', false);
      });
    }

    /* camera is the default mode on load */
    enterCamera();
  }

  document.addEventListener('DOMContentLoaded', init);
})();
