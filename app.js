"use strict";
/* AR indoor navigation - browser side.
 * Camera -> JPEG frames -> Flask (/api/process) -> detections + guidance.
 * Positioning = pedestrian dead reckoning (step counter + gyro heading), lightly
 * cross-checked against visual-odometry yaw from the server. See README limitations. */

const CFG = { targetLong: 640, minInterval: 110, arrivedDist: 1.2, camHeight: 1.4 };
const $ = s => document.querySelector(s);
const D2R = Math.PI / 180;
const wrap180 = a => { a = (a + 180) % 360; if (a < 0) a += 360; return a - 180; };
const wrap360 = a => ((a % 360) + 360) % 360;
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const sleep = ms => new Promise(r => setTimeout(r, ms));

const video = $('#video'), ov = $('#overlay'), octx = ov.getContext('2d');
const mini = $('#mini'), mctx = mini.getContext('2d');
const grab = document.createElement('canvas'), gctx = grab.getContext('2d');

const S = {
  sid: (crypto.randomUUID ? crypto.randomUUID() : String(Math.random()).slice(2)),
  running: false, voice: true,
  hs: null, haveOri: false, h0: 0, hBias: 0, pitchDown: 0,
  pos: { x: 0, y: 0 }, trail: [{ x: 0, y: 0 }], steps: 0, stepLen: 0.65,
  places: [], target: null,
  last: null, lastTs: 0, fps: 0, steer: 0, vo: null, voTs: 0,
  voSign: 1, voCorr: 0, voN: 0, prevCapHs: null,
  accel: { g: 9.8, lp: 0, last: 0, above: false },
  spoken: {}, lastSpeech: 0,
};
const START = { id: 'start', name: 'نقطة البداية', x: 0, y: 0, builtin: true };

/* ---------------- sensors ---------------- */
function headingRel() { return wrap180((S.hs || 0) + S.hBias - S.h0); }

function onOri(e) {
  if (e.alpha == null || e.beta == null || e.gamma == null) return;
  const a = e.alpha * D2R, b = e.beta * D2R, g = e.gamma * D2R;
  // world direction of the REAR camera axis (-z of the device), W3C Z-X'-Y'' rotation order
  const east = -(Math.cos(a) * Math.sin(g) + Math.sin(a) * Math.sin(b) * Math.cos(g));
  const north = -(Math.sin(a) * Math.sin(g) - Math.cos(a) * Math.sin(b) * Math.cos(g));
  S.hs = wrap360(Math.atan2(east, north) / D2R);
  S.pitchDown = Math.asin(clamp(Math.cos(b) * Math.cos(g), -1, 1)) / D2R;
  if (!S.haveOri) { S.haveOri = true; S.h0 = S.hs; }
}

function onMotion(e) {
  const a = e.accelerationIncludingGravity;
  if (!a || a.x == null) return;
  const m = Math.hypot(a.x, a.y, a.z), A = S.accel;
  A.g = A.g * 0.995 + m * 0.005;
  A.lp = A.lp * 0.6 + (m - A.g) * 0.4;
  const t = performance.now();
  if (!A.above && A.lp > 1.2 && t - A.last > 330) { A.above = true; A.last = t; registerStep(); }
  else if (A.above && A.lp < 0.3) A.above = false;
}

function registerStep(force) {
  if (!S.running || !S.haveOri) return;
  // reject "steps" while the camera sees no motion at all (hand shaking while standing still)
  if (!force && S.vo && S.vo.ok && !S.vo.moving && Date.now() - S.voTs < 800) return;
  const h = headingRel() * D2R;
  S.pos.x += S.stepLen * Math.sin(h);
  S.pos.y += S.stepLen * Math.cos(h);
  S.steps++;
  S.trail.push({ x: S.pos.x, y: S.pos.y });
  if (S.trail.length > 2000) S.trail.shift();
  updateInfo();
}

async function requestSensors() {
  try {
    for (const Cls of [window.DeviceOrientationEvent, window.DeviceMotionEvent]) {
      if (Cls && typeof Cls.requestPermission === 'function') {
        const r = await Cls.requestPermission();
        if (r !== 'granted') throw new Error('denied');
      }
    }
  } catch (e) { /* continue without sensors; chip will warn */ }
  window.addEventListener('deviceorientation', onOri, true);
  window.addEventListener('devicemotion', onMotion, true);
}

/* ---------------- visual odometry fusion ---------------- */
function fuseVO(vo, capHs) {
  if (capHs == null) return;
  if (S.prevCapHs != null && vo && vo.ok) {
    const dS = wrap180(capHs - S.prevCapHs);
    if (Math.abs(dS) > 1.5 && Math.abs(vo.yaw) < 30) {       // learn VO yaw sign vs gyro sign
      S.voCorr += Math.sign(dS) * Math.sign(vo.yaw) * Math.min(Math.abs(dS), 10); S.voN++;
    }
    if (S.voN >= 15) {
      S.voSign = S.voCorr >= 0 ? 1 : -1;
      const dis = vo.yaw * S.voSign - dS;
      if (vo.inliers >= 40 && Math.abs(dis) < 10) S.hBias += 0.12 * dis; // gentle correction
    }
  }
  S.prevCapHs = capHs;
}

/* ---------------- guidance ---------------- */
function waypointGuidance() {
  if (!S.target) return null;
  const dx = S.target.x - S.pos.x, dy = S.target.y - S.pos.y, dist = Math.hypot(dx, dy);
  if (dist < CFG.arrivedDist) return { code: 'arrived', text: 'وصلت إلى ' + S.target.name, hint: '', level: 'ok', steer: 0, done: true };
  const rel = wrap180(Math.atan2(dx, dy) / D2R - headingRel()), ar = Math.abs(rel);
  let text;
  if (ar < 20) text = 'امشِ للأمام';
  else if (ar < 60) text = rel > 0 ? 'انحرف يمين قليلًا' : 'انحرف يسار قليلًا';
  else if (ar < 150) text = rel > 0 ? 'لف يمين' : 'لف يسار';
  else text = 'لف للخلف';
  return { code: 'wp', text, hint: `${S.target.name}: ${dist.toFixed(1)} متر`, level: 'info', steer: clamp(rel, -60, 60) };
}

function decide(g) {
  let d = { ...g.default, steer: g.steer };
  const safety = ['obstacle', 'stairs', 'wall'].includes(d.code);
  if (!safety) {
    const w = waypointGuidance();
    if (w) { d = w; if (w.done) { S.target = null; renderPlaces(); } }
  }
  return d;
}

/* ---------------- voice ---------------- */
let arVoice = null;
function pickVoice() {
  if (!('speechSynthesis' in window)) return;
  arVoice = speechSynthesis.getVoices().find(v => v.lang && v.lang.toLowerCase().startsWith('ar')) || null;
}
if ('speechSynthesis' in window) { speechSynthesis.onvoiceschanged = pickVoice; pickVoice(); }

function speak(text, level) {
  if (!S.voice || !('speechSynthesis' in window) || !text) return;
  const now = Date.now();
  const cool = level === 'danger' ? 2500 : level === 'warn' ? 4000 : 7000;
  if (S.spoken[text] && now - S.spoken[text] < cool) return;
  if (level !== 'danger' && now - S.lastSpeech < 1500) return;
  if (level === 'danger') speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text);
  u.lang = 'ar-SA'; if (arVoice) u.voice = arVoice; u.rate = 1.0;
  speechSynthesis.speak(u);
  S.spoken[text] = now; S.lastSpeech = now;
}

/* ---------------- network loop ---------------- */
function setChip(id, text, cls) { const c = $('#' + id); c.textContent = text; c.className = 'chip ' + (cls || ''); }

function onResult(r, capHs) {
  S.last = r; S.lastTs = Date.now(); S.vo = r.vo; S.voTs = Date.now();
  fuseVO(r.vo, capHs);
  const d = decide(r.guidance);
  S.decision = d;
  $('#banner').className = 'banner ' + d.level;
  $('#bannerMain').textContent = d.text;
  $('#bannerHint').textContent = d.hint || '';
  speak(d.hint && d.level !== 'ok' && d.level !== 'info' ? `${d.text}، ${d.hint}` : d.text, d.level);
  setChip('chipNet', `${S.fps.toFixed(1)} إطار/ث · ${r.ms}ms`, 'good');
  if (r.vo.ok) setChip('chipVO', `تتبع بصري ✓ (${r.vo.inliers})`, 'good');
  else setChip('chipVO', 'تتبع بصري ضعيف', 'bad');
  updateInfo();
}

async function loop() {
  while (S.running) {
    const t0 = performance.now();
    const vw = video.videoWidth, vh = video.videoHeight;
    if (!vw) { await sleep(100); continue; }
    const sc = Math.min(1, CFG.targetLong / Math.max(vw, vh));
    const w = Math.round(vw * sc), h = Math.round(vh * sc);
    if (grab.width !== w || grab.height !== h) { grab.width = w; grab.height = h; }
    gctx.drawImage(video, 0, 0, w, h);
    const capHs = S.hs;
    const blob = await new Promise(r => grab.toBlob(r, 'image/jpeg', 0.62));
    try {
      const res = await fetch('/api/process', { method: 'POST', headers: { 'Content-Type': 'image/jpeg', 'X-Session': S.sid }, body: blob });
      if (!res.ok) throw new Error('HTTP ' + res.status);
      onResult(await res.json(), capHs);
    } catch (err) {
      setChip('chipNet', 'انقطع الاتصال بالخادم', 'bad');
      $('#bannerMain').textContent = 'انقطع الاتصال بالخادم'; $('#bannerHint').textContent = 'تأكد من الواي فاي وأن الخادم يعمل';
      $('#banner').className = 'banner warn';
      await sleep(800);
    }
    const dt = performance.now() - t0;
    S.fps = S.fps * 0.8 + (1000 / dt) * 0.2;
    if (dt < CFG.minInterval) await sleep(CFG.minInterval - dt);
  }
}

/* ---------------- AR overlay ---------------- */
const KIND_COLOR = { obstacle: '#ff5a5f', door: '#4cc9f0', stairs: '#ffb347', wall: '#b388ff' };
const LEVEL_COLOR = { ok: '#3be8a5', info: '#4cc9f0', warn: '#ffb347', danger: '#ff5a5f' };

function project(X, Z, W, H, f, p) {
  const yc = CFG.camHeight * Math.cos(p) - Z * Math.sin(p);
  const zc = CFG.camHeight * Math.sin(p) + Z * Math.cos(p);
  if (zc < 0.25) return null;
  return [W / 2 + f * X / zc, H / 2 + f * yc / zc];
}

function drawArrow(W, H, f, steerDeg, color) {
  const th = steerDeg * D2R, p = clamp(S.pitchDown, -45, 70) * D2R;
  const shape = [[-.13, 1.2], [.13, 1.2], [.13, 2.6], [.38, 2.6], [0, 3.6], [-.38, 2.6], [-.13, 2.6]];
  const pts = [];
  for (const [u, v] of shape) {
    const X = u * Math.cos(th) + v * Math.sin(th), Z = v * Math.cos(th) - u * Math.sin(th);
    const q = project(X, Z, W, H, f, p);
    if (!q) return;
    pts.push(q);
  }
  octx.save();
  octx.globalAlpha = 0.62 + 0.18 * Math.sin(performance.now() / 260);
  octx.fillStyle = color; octx.strokeStyle = '#fff'; octx.lineWidth = 2.5; octx.lineJoin = 'round';
  octx.shadowColor = color; octx.shadowBlur = 18;
  octx.beginPath(); pts.forEach(([x, y], i) => i ? octx.lineTo(x, y) : octx.moveTo(x, y)); octx.closePath();
  octx.fill(); octx.stroke(); octx.restore();
}

function render() {
  requestAnimationFrame(render);
  const W = ov.clientWidth, H = ov.clientHeight;
  if (!W || !H) return;
  const dpr = window.devicePixelRatio || 1;
  if (ov.width !== Math.round(W * dpr) || ov.height !== Math.round(H * dpr)) { ov.width = Math.round(W * dpr); ov.height = Math.round(H * dpr); }
  octx.setTransform(dpr, 0, 0, dpr, 0, 0); octx.clearRect(0, 0, W, H);
  const r = S.last;
  if (!r || Date.now() - S.lastTs > 2500) return;
  const f = (H > W ? 0.95 : 0.80) * W;

  octx.font = '600 14px "Segoe UI","Noto Naskh Arabic",Tahoma,sans-serif';
  octx.direction = 'rtl'; octx.textBaseline = 'middle'; octx.textAlign = 'left';
  for (const d of r.detections) {
    const [x1, y1, x2, y2] = d.box, c = KIND_COLOR[d.kind] || '#fff';
    const X = x1 * W, Y = y1 * H, BW = (x2 - x1) * W, BH = (y2 - y1) * H;
    octx.lineWidth = 3; octx.strokeStyle = c; octx.setLineDash(d.src === 'cv' ? [8, 6] : []);
    octx.strokeRect(X, Y, BW, BH); octx.setLineDash([]);
    const label = `${d.label} ${Math.round(d.conf * 100)}%`, tw = octx.measureText(label).width + 12;
    const ly = Y > 74 ? Y - 22 : Y + 2;               // keep labels clear of the top banner
    octx.fillStyle = c; octx.fillRect(X, ly, tw, 22);
    octx.fillStyle = '#08161a'; octx.fillText(label, X + 6, ly + 11);
  }
  if (r.vp && r.vp_conf >= 0.4) {
    const x = r.vp[0] * W, y = r.vp[1] * H;
    octx.strokeStyle = 'rgba(59,232,165,.8)'; octx.lineWidth = 2;
    octx.beginPath(); octx.arc(x, y, 9, 0, 7); octx.moveTo(x - 16, y); octx.lineTo(x + 16, y);
    octx.moveTo(x, y - 16); octx.lineTo(x, y + 16); octx.stroke();
    octx.fillStyle = '#3be8a5'; octx.fillText('الممر', x + 14, y - 14);
  }
  const d = S.decision;
  if (d) {
    S.steer += (d.steer - S.steer) * 0.2;
    drawArrow(W, H, f, S.steer, LEVEL_COLOR[d.level] || '#3be8a5');
  }
}

/* ---------------- mini map / info ---------------- */
function drawMini() {
  const W = mini.width, H = mini.height;
  mctx.clearRect(0, 0, W, H);
  const pts = [...S.trail, ...S.places, START];
  let maxd = 6;
  for (const p of pts) maxd = Math.max(maxd, Math.abs(p.x - S.pos.x), Math.abs(p.y - S.pos.y));
  const sc = (W / 2 - 18) / maxd, X = p => W / 2 + (p.x - S.pos.x) * sc, Y = p => H / 2 - (p.y - S.pos.y) * sc;
  mctx.strokeStyle = '#1f3c44'; mctx.lineWidth = 1;
  for (const m of [maxd / 2, maxd]) { mctx.beginPath(); mctx.arc(W / 2, H / 2, m * sc, 0, 7); mctx.stroke(); }
  mctx.strokeStyle = '#3be8a5'; mctx.lineWidth = 2; mctx.beginPath();
  S.trail.forEach((p, i) => i ? mctx.lineTo(X(p), Y(p)) : mctx.moveTo(X(p), Y(p))); mctx.stroke();
  mctx.font = '13px "Segoe UI",Tahoma,sans-serif'; mctx.direction = 'rtl';
  for (const p of [START, ...S.places]) {
    const act = S.target && S.target.id === p.id;
    mctx.fillStyle = act ? '#ffb347' : '#4cc9f0';
    mctx.beginPath(); mctx.arc(X(p), Y(p), 5, 0, 7); mctx.fill();
    mctx.fillText(p.name, X(p) + 8, Y(p) - 6);
  }
  if (S.target) {
    mctx.setLineDash([5, 5]); mctx.strokeStyle = '#ffb347'; mctx.beginPath();
    mctx.moveTo(W / 2, H / 2); mctx.lineTo(X(S.target), Y(S.target)); mctx.stroke(); mctx.setLineDash([]);
  }
  mctx.save(); mctx.translate(W / 2, H / 2); mctx.rotate(headingRel() * D2R);
  mctx.fillStyle = '#fff'; mctx.beginPath(); mctx.moveTo(0, -11); mctx.lineTo(7, 8); mctx.lineTo(0, 4); mctx.lineTo(-7, 8); mctx.closePath(); mctx.fill();
  mctx.restore();
}

function updateInfo() {
  $('#stSteps').textContent = S.steps;
  $('#stPos').textContent = `${S.pos.x.toFixed(1)} , ${S.pos.y.toFixed(1)}`;
  $('#stHead').textContent = S.haveOri ? Math.round(wrap360(headingRel())) + '°' : 'لا توجد حساسات';
  if (S.target) {
    const dist = Math.hypot(S.target.x - S.pos.x, S.target.y - S.pos.y);
    $('#stTarget').textContent = `${S.target.name} (${dist.toFixed(1)} م)`;
  } else $('#stTarget').textContent = 'لا يوجد';
  drawMini();
}

/* ---------------- places ---------------- */
function persist() { try { localStorage.setItem('arnav_places', JSON.stringify(S.places)); } catch (e) { /* private mode */ } }
function loadPlaces() { try { S.places = JSON.parse(localStorage.getItem('arnav_places') || '[]'); } catch (e) { S.places = []; } }

function renderPlaces() {
  const ul = $('#places'); ul.innerHTML = '';
  for (const p of [START, ...S.places]) {
    const li = document.createElement('li');
    if (S.target && S.target.id === p.id) li.className = 'active';
    const dist = Math.hypot(p.x - S.pos.x, p.y - S.pos.y);
    const name = document.createElement('span'); name.className = 'name'; name.textContent = p.name;
    const sm = document.createElement('small'); sm.textContent = dist.toFixed(1) + ' م';
    const go = document.createElement('button');
    const active = S.target && S.target.id === p.id;
    go.textContent = active ? 'إلغاء' : 'اذهب';
    go.onclick = () => { S.target = active ? null : p; renderPlaces(); updateInfo(); };
    li.append(name, sm, go);
    if (!p.builtin) {
      const del = document.createElement('button'); del.textContent = 'حذف';
      del.onclick = () => { S.places = S.places.filter(q => q.id !== p.id); if (S.target && S.target.id === p.id) S.target = null; persist(); renderPlaces(); updateInfo(); };
      li.append(del);
    }
    ul.append(li);
  }
}

/* ---------------- start-up ---------------- */
async function startCamera() {
  if (!window.isSecureContext || !navigator.mediaDevices) throw new Error('insecure');
  const tries = [
    { video: { facingMode: { exact: 'environment' }, width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false },
    { video: { facingMode: 'environment' }, audio: false },
  ];
  let stream = null, lastErr = null;
  for (const c of tries) { try { stream = await navigator.mediaDevices.getUserMedia(c); break; } catch (e) { lastErr = e; } }
  if (!stream) throw lastErr || new Error('nocam');
  video.srcObject = stream; await video.play();
  const st = stream.getVideoTracks()[0].getSettings();
  const rear = st.facingMode ? st.facingMode === 'environment' : /back|rear|environment|خلف/i.test(stream.getVideoTracks()[0].label);
  setChip('chipCam', rear ? 'الكاميرا الخلفية ✓' : 'تحذير: قد لا تكون خلفية', rear ? 'good' : 'bad');
}

function errMessage(e) {
  const n = e && e.name;
  if (e && e.message === 'insecure') return 'الكاميرا تحتاج اتصال HTTPS. افتح العنوان الذي يبدأ بـ https:// ووافق على تحذير الشهادة.';
  if (n === 'NotAllowedError' || n === 'SecurityError') return 'تم رفض إذن الكاميرا. اسمح بالكاميرا من إعدادات المتصفح ثم أعد تحميل الصفحة.';
  if (n === 'NotFoundError' || n === 'OverconstrainedError') return 'لم يتم العثور على كاميرا خلفية على هذا الجهاز.';
  if (n === 'NotReadableError') return 'الكاميرا مستخدمة من تطبيق آخر. أغلقه وحاول مرة أخرى.';
  return 'تعذر تشغيل الكاميرا: ' + (e && (e.message || e.name) || 'خطأ غير معروف');
}

$('#btnStart').onclick = async () => {
  $('#err').textContent = '';
  try {
    await requestSensors();                 // must run inside the tap (iOS permission rule)
    await startCamera();
  } catch (e) { $('#err').textContent = errMessage(e); return; }
  $('#intro').hidden = true; $('#stage').hidden = false;
  loadPlaces(); renderPlaces(); updateInfo();
  S.running = true;
  if ('speechSynthesis' in window) {
    speak('تم تفعيل الإرشاد الصوتي', 'info');
    setTimeout(() => { if (!arVoice) { pickVoice(); if (!arVoice) $('#bannerHint').textContent = 'لا يوجد صوت عربي على هذا الجهاز: الإرشاد نصي فقط'; } }, 1200);
  }
  setTimeout(() => { if (!S.haveOri) setChip('chipVO', 'حساسات الحركة غير متاحة: التتبع معطّل', 'bad'); }, 3000);
  render(); loop();
};

$('#btnVoice').onclick = e => {
  S.voice = !S.voice; if (!S.voice && 'speechSynthesis' in window) speechSynthesis.cancel();
  e.target.textContent = S.voice ? '🔊 الصوت: يعمل' : '🔇 الصوت: متوقف'; e.target.classList.toggle('on', S.voice);
};
$('#btnSave').onclick = () => {
  const name = prompt('اسم المكان (مثال: باب المكتب)');
  if (!name) return;
  S.places.push({ id: 'p' + Date.now(), name: name.trim(), x: S.pos.x, y: S.pos.y });
  persist(); renderPlaces(); updateInfo();
};
$('#btnReset').onclick = () => {
  S.pos = { x: 0, y: 0 }; S.trail = [{ x: 0, y: 0 }]; S.steps = 0; S.target = null;
  S.h0 = S.hs || 0; S.hBias = 0; S.prevCapHs = null;
  fetch('/api/reset', { method: 'POST', headers: { 'X-Session': S.sid } }).catch(() => {});
  renderPlaces(); updateInfo();
};
$('#btnStep').onclick = () => registerStep(true);
$('#stepLen').onchange = e => { S.stepLen = clamp(parseFloat(e.target.value) || 0.65, 0.4, 1.0); };
document.addEventListener('visibilitychange', () => { if (document.hidden && 'speechSynthesis' in window) speechSynthesis.cancel(); });
