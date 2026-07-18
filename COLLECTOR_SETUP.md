# Collector setup — receive tester photos in Google Drive + Sheet

The app auto-sends every scan (photo + annotated result + verdict) to you the
moment a tester scans, and updates the row with their ✓/✗ correctness a moment
later. Nothing is sent until you do this **one-time** setup and paste the URL
into `static/collector-config.js`. Until then the app still works and testers can
hand you the **DOWNLOAD ALL (ZIP)** file instead.

Takes about 5 minutes.

## 1. Make a Drive folder and a Google Sheet

1. In Google Drive, create a folder, e.g. **`glasses-tests`**. Open it and copy the
   ID from the URL: `https://drive.google.com/drive/folders/`**`<FOLDER_ID>`**.
2. Create a new Google Sheet, e.g. **`glasses-tests`**. Copy its ID from the URL:
   `https://docs.google.com/spreadsheets/d/`**`<SHEET_ID>`**`/edit`.

## 2. Add the Apps Script

1. In the Sheet: **Extensions → Apps Script**.
2. Delete any starter code and paste everything below.
3. Fill in `FOLDER_ID` and `SHEET_ID` at the top.

```javascript
// ==== CONFIG (already filled in for your Drive folder + Sheet) ====
var FOLDER_ID = '1TdaRaCkRSlYij7rGP82D8eZi8LAUSYH0';
var SHEET_ID  = '1F9pq-c9ATvDdo0wOzkFSavhTLh6lSgZBINKpGyzM5wE';
var SHEET_TAB = 'Scans';

function doPost(e) {
  try {
    var data = JSON.parse(e.postData.contents);   // body is text/plain JSON
    var sh = getSheet_();

    if (data.action === 'tag') {
      // follow-up: write the tester's correctness onto the existing row
      updateCorrect_(sh, data.id, data.correct);
      return json_({ status: 'ok' });
    }

    // action === 'scan' (default): save the images + append a row
    var folder = DriveApp.getFolderById(FOLDER_ID);
    var safeName = String(data.name || 'anon').replace(/[^\w\-]+/g, '_');
    var base = safeName + '_' + new Date().getTime();
    var photoUrl  = data.photo   ? saveDataUri_(folder, data.photo,   base + '_photo')  : '';
    var resultUrl = data.overlay ? saveDataUri_(folder, data.overlay, base + '_result') : '';

    sh.appendRow([
      data.id || '',
      data.timestamp || new Date().toISOString(),
      data.name || '',
      data.verdict || '',
      data.confidence || '',
      '',                 // correct — filled later by the 'tag' follow-up
      photoUrl,
      resultUrl
    ]);
    return json_({ status: 'ok', photo: photoUrl, result: resultUrl });
  } catch (err) {
    return json_({ status: 'error', message: String(err) });
  }
}

function getSheet_() {
  var ss = SpreadsheetApp.openById(SHEET_ID);
  var sh = ss.getSheetByName(SHEET_TAB) || ss.insertSheet(SHEET_TAB);
  if (sh.getLastRow() === 0) {
    sh.appendRow(['id', 'timestamp', 'name', 'verdict', 'confidence',
                  'correct', 'photo_link', 'result_link']);
  }
  return sh;
}

function updateCorrect_(sh, id, correct) {
  if (!id) { return; }
  var ids = sh.getRange(1, 1, sh.getLastRow(), 1).getValues();   // column A = id
  for (var r = ids.length - 1; r >= 1; r -= 1) {                 // newest first, skip header
    if (String(ids[r][0]) === String(id)) {
      sh.getRange(r + 1, 6).setValue(correct);                   // column F = correct
      return;
    }
  }
  // no matching row yet (tag beat its scan, or the scan write failed) — don't
  // drop the correctness; append a row keyed by id so it is never lost.
  sh.appendRow([id, new Date().toISOString(), '', '', '', correct, '', '']);
}

function saveDataUri_(folder, dataUri, filename) {
  var m = /^data:([^;]+);base64,(.*)$/.exec(dataUri);
  var mime = m ? m[1] : 'image/jpeg';
  var b64  = m ? m[2] : dataUri;
  var ext  = mime.indexOf('png') !== -1 ? '.png' : '.jpg';
  var blob = Utilities.newBlob(Utilities.base64Decode(b64), mime, filename + ext);
  var file = folder.createFile(blob);
  file.setSharing(DriveApp.Access.ANYONE_WITH_LINK, DriveApp.Permission.VIEW);
  return file.getUrl();
}

function json_(obj) {
  return ContentService.createTextOutput(JSON.stringify(obj))
    .setMimeType(ContentService.MimeType.JSON);
}

function doGet() { return json_({ status: 'ok', hint: 'POST scans here' }); }
```

> `setSharing(ANYONE_WITH_LINK, VIEW)` makes the `photo_link`/`result_link`
> columns clickable. Remove that line if you'd rather keep the files private
> (the links will then only open for accounts with access to the folder).

## 3. Deploy as a Web App

1. **Deploy → New deployment**.
2. Click the gear → **Web app**.
3. **Execute as:** `Me`.
4. **Who has access:** `Anyone`  ← required so the site can POST without a login.
   (Do *not* pick "Anyone with Google account" — that blocks anonymous testers.)
5. **Deploy**, then **Authorize access** and grant the Drive + Sheets permissions.
6. Copy the **Web app URL** (it ends in `/exec`).

## 4. Turn it on in the site

Open `static/collector-config.js` and paste the URL:

```js
window.COLLECTOR_URL = 'https://script.google.com/macros/s/AKfy.../exec';
```

Commit + deploy (or just save for local testing). Done — each scan now lands in
your Drive folder, a row appears in the Sheet, and the ✓/✗ correctness fills in
on that same row when the tester taps it.

## Changing the script later

Edit the code, then **Deploy → Manage deployments → (edit / pencil) → New
version → Deploy**. The `/exec` URL stays the same. (Creating a brand-new
deployment instead would mint a *different* URL you'd have to re-paste.)

## Notes / limits

- Apps Script free quotas are generous but not unlimited (daily runtime,
  6-minutes-per-call, Drive storage). This endpoint is for low-volume tester
  feedback; the photos are downscaled JPEGs to keep each send small.
- Every scan is also kept in the browser session — **DOWNLOAD ALL (ZIP)** is the
  bulk hand-off and a backstop if a send ever fails.
