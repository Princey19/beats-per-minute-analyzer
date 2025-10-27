// ======= Utilities =======

      const $ = (id) => document.getElementById(id);
      const qs = (s) => document.querySelector(s);
      const fmt = (n, d = 0) => (Number.isFinite(n) ? n.toFixed(d) : "—");

      function drawLine(canvas, values) {
        const ctx = canvas.getContext("2d");
        const W = canvas.width,
          H = canvas.height;
        ctx.clearRect(0, 0, W, H);
        ctx.lineWidth = 2;
        ctx.strokeStyle = "#6ecbff";
        ctx.beginPath();
        const N = values.length;
        if (!N) {
          ctx.stroke();
          return;
        }
        for (let i = 0; i < N; i++) {
          const x = (i / (N - 1)) * W;
          const y = (1 - values[i]) * H; // invert so 1.0 is top
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.stroke();
      }

      function drawTempoCurve(canvas, points, minBPM, maxBPM) {
        const ctx = canvas.getContext("2d");
        const W = canvas.width,
          H = canvas.height;
        ctx.clearRect(0, 0, W, H);
        if (!points.length) {
          return;
        }
        const tMin = points[0].time,
          tMax = points[points.length - 1].time;
        const toX = (t) => ((t - tMin) / Math.max(1e-6, tMax - tMin)) * W;
        const toY = (bpm) =>
          H - ((bpm - minBPM) / Math.max(1e-6, maxBPM - minBPM)) * H;
        ctx.lineWidth = 2;
        ctx.strokeStyle = "#7bd88f";
        ctx.beginPath();
        for (let i = 0; i < points.length; i++) {
          const x = toX(points[i].time);
          const y = toY(points[i].bpm);
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.stroke();
        // Axis labels (simple)
        ctx.fillStyle = "#9fb3c8";
        ctx.font = "12px system-ui, sans-serif";
        ctx.fillText(minBPM + " BPM", 6, H - 6);
        ctx.fillText(maxBPM + " BPM", 6, 12);
      }

      // ======= Audio loading helpers =======
      let audioCtx; // created lazily
      let loadedBuffer = null;

      async function ensureAudioContext() {
        if (!audioCtx) {
          const AC = window.AudioContext || window.webkitAudioContext;
          audioCtx = new AC();
        }
        if (audioCtx.state === "suspended") await audioCtx.resume();
        return audioCtx;
      }

      async function decodeArrayBuffer(arrBuf) {
        const ctx = await ensureAudioContext();
        return new Promise((resolve, reject) => {
          ctx.decodeAudioData(arrBuf.slice(0), resolve, reject);
        });
      }

      function generateDemoBuffer() {
        const sr = 44100;
        const seconds = 16;
        const bpm = 120; // demo tempo
        const frames = sr * seconds;
        const channels = 1;
        const buf = audioCtx.createBuffer(channels, frames, sr);
        const data = buf.getChannelData(0);
        const interval = Math.round((60 / bpm) * sr);
        for (let i = 0; i < frames; i++) {
          const isClick = i % interval < 200; // short click
          const env = isClick ? Math.exp(-((i % interval) / 200)) : 0;
          data[i] = env * Math.sin(2 * Math.PI * 1000 * (i / sr));
        }
        return buf;
      }

      function sliceBuffer(buffer, startFrame, lengthFrames) {
        const sr = buffer.sampleRate;
        const channels = buffer.numberOfChannels;
        const out = audioCtx.createBuffer(channels, lengthFrames, sr);
        for (let c = 0; c < channels; c++) {
          const src = buffer.getChannelData(c);
          const dst = out.getChannelData(c);
          for (let i = 0; i < lengthFrames; i++)
            dst[i] = src[startFrame + i] || 0;
        }
        return out;
      }

      // ======= Core BPM estimation =======
      function estimateBPM(audioBuffer, opts = {}) {
        const cfg = Object.assign(
          {
            frameSize: 1024,
            hopSize: 512,
            minBPM: 60,
            maxBPM: 180,
            smoothWin: 3,
            preEmphasis: false,
          },
          opts
        );
        const sr = audioBuffer.sampleRate;
        const ch = audioBuffer.numberOfChannels;
        const N = audioBuffer.length;
        const mono = new Float32Array(N);
        for (let c = 0; c < ch; c++) {
          const d = audioBuffer.getChannelData(c);
          for (let i = 0; i < N; i++) mono[i] += d[i] / ch;
        }
        if (cfg.preEmphasis) {
          let prev = 0;
          for (let i = 0; i < N; i++) {
            const x = mono[i];
            mono[i] = mono[i] - 0.97 * prev;
            prev = x;
          }
        }
        const { frameSize, hopSize } = cfg;
        const frames = 1 + Math.max(0, Math.floor((N - frameSize) / hopSize));
        const energy = new Float32Array(frames);
        let pos = 0;
        for (let f = 0; f < frames; f++) {
          let sumsq = 0;
          for (let i = 0; i < frameSize; i++) {
            const s = mono[pos + i] || 0;
            sumsq += s * s;
          }
          energy[f] = sumsq / frameSize;
          pos += hopSize;
        }
        const onset = new Float32Array(frames);
        onset[0] = 0;
        for (let i = 1; i < frames; i++) {
          const diff = energy[i] - energy[i - 1];
          onset[i] = diff > 0 ? diff : 0;
        }
        const sm = movingAverage(onset, cfg.smoothWin);
        const zn = zNormalize(sm);
        const env = toUnitRange(zn);
        const envRate = sr / hopSize;
        const [minLag, maxLag] = bpmToLagRange(cfg.minBPM, cfg.maxBPM, envRate);
        const acf = autocorrelate(env, minLag, maxLag);
        const peaks = findTopPeaks(acf, 5, 2);
        const bpmCandidates = peaks
          .map((p) => ({
            bpm: lagToBPM(minLag + p.index, envRate),
            strength: p.value,
          }))
          .sort((a, b) => b.strength - a.strength);
        let best = bpmCandidates[0] || { bpm: NaN, strength: 0 };
        if (Number.isFinite(best.bpm))
          best.bpm = refineTempo(
            best.bpm,
            bpmCandidates.map((c) => c.bpm)
          );
        const next = bpmCandidates[1]?.strength || 1e-6;
        const conf = best.strength ? best.strength / (next + 1e-6) : 0;
        return {
          bpm: best.bpm,
          confidence: conf,
          envelope: env,
          acf: acf,
          bpmCandidates,
        };
      }

      function estimateBPMOverTime(buffer, opts = {}) {
        const cfg = Object.assign(
          {
            winSecs: 8,
            hopSecs: 2,
            frameSize: 1024,
            hopSize: 512,
            minBPM: 60,
            maxBPM: 180,
          },
          opts
        );
        const sr = buffer.sampleRate;
        const win = Math.floor(cfg.winSecs * sr);
        const hop = Math.floor(cfg.hopSecs * sr);
        const points = [];
        if (buffer.length < win) {
          const r = estimateBPM(buffer, cfg);
          const mid = buffer.length / sr / 2;
          if (Number.isFinite(r.bpm))
            points.push({ time: mid, bpm: r.bpm, conf: r.confidence });
          return points;
        }
        for (let start = 0; start + win <= buffer.length; start += hop) {
          const seg = sliceBuffer(buffer, start, win);
          const r = estimateBPM(seg, cfg);
          const mid = (start + win / 2) / sr;
          if (Number.isFinite(r.bpm))
            points.push({ time: mid, bpm: r.bpm, conf: r.confidence });
        }
        return points;
      }

      function movingAverage(arr, win) {
        if (win <= 1) return Float32Array.from(arr);
        const N = arr.length,
          out = new Float32Array(N);
        let acc = 0;
        for (let i = 0; i < N; i++) {
          acc += arr[i];
          if (i >= win) acc -= arr[i - win];
          out[i] = acc / Math.min(win, i + 1);
        }
        return out;
      }
      function zNormalize(arr) {
        const N = arr.length;
        let sum = 0,
          sum2 = 0;
        for (let i = 0; i < N; i++) {
          sum += arr[i];
          sum2 += arr[i] * arr[i];
        }
        const mean = sum / N;
        const varr = Math.max(1e-12, sum2 / N - mean * mean);
        const std = Math.sqrt(varr);
        const out = new Float32Array(N);
        for (let i = 0; i < N; i++) out[i] = (arr[i] - mean) / std;
        return out;
      }
      function toUnitRange(arr) {
        let min = Infinity,
          max = -Infinity;
        for (const v of arr) {
          if (v < min) min = v;
          if (v > max) max = v;
        }
        const span = Math.max(1e-9, max - min);
        const out = new Float32Array(arr.length);
        for (let i = 0; i < arr.length; i++) out[i] = (arr[i] - min) / span;
        return out;
      }
      function bpmToLagRange(minBPM, maxBPM, rate) {
        const maxLag = Math.round((rate * 60) / minBPM);
        const minLag = Math.max(1, Math.round((rate * 60) / maxBPM));
        return [minLag, maxLag];
      }
      function lagToBPM(lag, rate) {
        return (60 * rate) / lag;
      }
      function autocorrelate(x, minLag, maxLag) {
        const N = x.length;
        let sum = 0;
        for (let i = 0; i < N; i++) sum += x[i];
        const mean = sum / N;
        const xm = new Float32Array(N);
        for (let i = 0; i < N; i++) xm[i] = x[i] - mean;
        const acLen = maxLag - minLag + 1;
        const acf = new Float32Array(acLen);
        for (let lag = minLag; lag <= maxLag; lag++) {
          let acc = 0;
          for (let n = 0; n < N - lag; n++) acc += xm[n] * xm[n + lag];
          acf[lag - minLag] = acc;
        }
        let acMin = Infinity,
          acMax = -Infinity;
        for (let i = 0; i < acLen; i++) {
          const v = acf[i];
          if (v < acMin) acMin = v;
          if (v > acMax) acMax = v;
        }
        const span = Math.max(1e-9, acMax - acMin);
        for (let i = 0; i < acLen; i++) acf[i] = (acf[i] - acMin) / span;
        return acf;
      }
      function findTopPeaks(series, k = 3, minDistance = 2) {
        const peaks = [];
        for (let i = 1; i < series.length - 1; i++) {
          if (series[i] > series[i - 1] && series[i] > series[i + 1])
            peaks.push({ index: i, value: series[i] });
        }
        peaks.sort((a, b) => b.value - a.value);
        const picked = [];
        for (const p of peaks) {
          if (picked.length >= k) break;
          if (picked.every((q) => Math.abs(q.index - p.index) >= minDistance))
            picked.push(p);
        }
        return picked;
      }
      function refineTempo(bestBPM, cands) {
        const prefs = [
          bestBPM,
          bestBPM / 2,
          bestBPM * 2,
          bestBPM / 3,
          bestBPM * 3,
        ];
        const inRange = prefs.filter((b) => b >= 60 && b <= 180);
        if (inRange.length) {
          let best = inRange[0],
            bestDist = Math.abs(inRange[0] - 120);
          for (const v of inRange) {
            const d = Math.abs(v - 120);
            if (d < bestDist) {
              best = v;
              bestDist = d;
            }
          }
          return best;
        }
        return Math.max(60, Math.min(180, bestBPM));
      }

      // ======= Tap Tempo =======
      let tapTimes = [];
      function registerTap() {
        const now = performance.now();
        const last = tapTimes[tapTimes.length - 1];
        tapTimes.push(now);
        if (last && now - last > 2000) {
          // big gap resets series
          tapTimes = [now];
        }
        if (tapTimes.length >= 4) {
          const intervals = [];
          for (let i = 1; i < tapTimes.length; i++) {
            intervals.push(tapTimes[i] - tapTimes[i - 1]);
          }
          // Remove outliers via simple median filter
          const sorted = intervals.slice().sort((a, b) => a - b);
          const med = sorted[Math.floor(sorted.length / 2)];
          const filtered = intervals.filter(
            (v) => Math.abs(v - med) < 0.2 * med
          );
          const avg = filtered.reduce((a, b) => a + b, 0) / filtered.length;
          const bpm = 60000 / avg;
          $("tapValue").textContent = Math.round(bpm) + " BPM";
        }
        $("tapCount").textContent = tapTimes.length + " taps";
      }
      function resetTap() {
        tapTimes = [];
        $("tapValue").textContent = "—";
        $("tapCount").textContent = "0 taps";
      }

      // ======= UI glue =======
      const fileInput = $("fileInput");
      const micBtn = $("micBtn");
      const analyzeBtn = $("analyzeBtn");
      const demoBtn = $("demoBtn");
      const exportCsvBtn = $("exportCsvBtn"); // exports tempo curve for lastCurve
      const downloadResultsBtn = $("downloadResultsBtn"); // exports summary CSV
      const statusEl = $("status");
      const metrics = $("metrics");
      const bpmValue = $("bpmValue");
      const bpmNote = $("bpmNote");
      const confValue = $("confValue");
      const envCanvas = $("envCanvas");
      const acfCanvas = $("acfCanvas");
      const tempoCanvas = $("tempoCanvas");
      const minBPMEl = $("minBPM");
      const maxBPMEl = $("maxBPM");
      const perWindowEl = $("perWindow");
      const winSecsEl = $("winSecs");
      const hopSecsEl = $("hopSecs");

      const progressText = $("progressText");
      const progressFill = $("progressFill");
      const resultsTableBody = $("resultsTable").querySelector("tbody");

      let lastCurve = [];
      let results = []; // { filename, duration, bpm, confidence, curve (optional) }

      // Handle file selection: save selected files to window.fileQueue
      fileInput.addEventListener("change", async (e) => {
        try {
          const files = Array.from(e.target.files || []);
          if (!files.length) {
            statusEl.textContent = "No files selected.";
            return;
          }
          window.fileQueue = files;
          statusEl.textContent = `Loaded ${files.length} file(s). Ready to analyze.`;
          analyzeBtn.disabled = false;
          downloadResultsBtn.disabled = true;
          exportCsvBtn.disabled = true;
          // clear previous results for a fresh run
          results = [];
          renderResultsTable();
          updateProgress(0, files.length);
        } catch (err) {
          console.error(err);
          statusEl.textContent = `Error loading files: ${err.message || err}`;
        }
      });

      micBtn.addEventListener("click", async () => {
        try {
          await ensureAudioContext();
          statusEl.textContent = "Requesting microphone…";
          const stream = await navigator.mediaDevices.getUserMedia({
            audio: true,
          });
          statusEl.textContent = "Recording 10 seconds…";
          const rec = new MediaRecorder(stream, {
            mimeType: "audio/webm;codecs=opus",
          });
          const chunks = [];
          rec.ondataavailable = (e) => {
            if (e.data && e.data.size > 0) chunks.push(e.data);
          };
          rec.onstop = async () => {
            const blob = new Blob(chunks, { type: "audio/webm;codecs=opus" });
            const arr = await blob.arrayBuffer();
            try {
              await ensureAudioContext();
              loadedBuffer = await decodeArrayBuffer(arr);
              statusEl.textContent = `Captured mic audio (${fmt(
                loadedBuffer.duration,
                1
              )}s @ ${loadedBuffer.sampleRate}Hz)`;
              analyzeBtn.disabled = false;
              exportCsvBtn.disabled = true;
              downloadResultsBtn.disabled = true;
            } catch (err) {
              statusEl.textContent =
                "Could not decode microphone audio. Try again or use a file.";
            }
            stream.getTracks().forEach((t) => t.stop());
          };
          rec.start();
          setTimeout(() => {
            if (rec.state !== "inactive") rec.stop();
          }, 10000);
        } catch (err) {
          console.error(err);
          statusEl.textContent = ` Mic error: ${err.message || err};`;
        }
      });

      demoBtn.addEventListener("click", async () => {
        await ensureAudioContext();
        loadedBuffer = generateDemoBuffer();
        statusEl.textContent = `Demo loaded: synthetic clicks at ~120 BPM (${fmt(
          loadedBuffer.duration,
          1
        )}s)`;
        analyzeBtn.disabled = false;
        exportCsvBtn.disabled = true;
        downloadResultsBtn.disabled = true;
      });

      [minBPMEl, maxBPMEl].forEach((el) =>
        el.addEventListener("change", () => {
          const min = Number(minBPMEl.value) || 60,
            max = Number(maxBPMEl.value) || 180;
          $("rangeValue").textContent = ` ${min}–${max}`;
        })
      );

      $("tapBtn").addEventListener("click", registerTap);
      $("tapResetBtn").addEventListener("click", resetTap);

      function renderResultsTable() {
        resultsTableBody.innerHTML = "";
        for (const r of results) {
          const tr = document.createElement("tr");
          const fn = document.createElement("td");
          fn.textContent = r.filename;
          const dur = document.createElement("td");
          dur.textContent = fmt(r.duration, 2);
          const bpm = document.createElement("td");
          bpm.textContent = Number.isFinite(r.bpm)
            ? Math.round(r.bpm) + " BPM"
            : "—";
          const conf = document.createElement("td");
          conf.textContent = fmt(r.confidence, 3);
          tr.appendChild(fn);
          tr.appendChild(dur);
          tr.appendChild(bpm);
          tr.appendChild(conf);
          resultsTableBody.appendChild(tr);
        }
      }

      function updateProgress(done, total) {
        if (!total || total <= 0) {
          progressText.textContent = "No analysis yet.";
          progressFill.style.width = "0%";
          return;
        }
        const pct = Math.round((done / total) * 100);
        progressText.textContent = `Analyzing ${done} of ${total}`;
        progressFill.style.width = pct + "%";
      }

      // Analyze all files sequentially
      analyzeBtn.addEventListener("click", async () => {
        const files = window.fileQueue || [];
        if (!files.length && !loadedBuffer) {
          statusEl.textContent = "Please select files or record audio first.";
          return;
        }

        analyzeBtn.disabled = true;
        downloadResultsBtn.disabled = true;
        exportCsvBtn.disabled = true;

        // If there are selected files, process them sequentially.
        if (files.length) {
          results = [];
          renderResultsTable();
          updateProgress(0, files.length);

          for (let i = 0; i < files.length; i++) {
            const file = files[i];
            statusEl.textContent = `Reading ${file.name}… (${i + 1}/${
              files.length
            })`;
            try {
              const arrBuf = await file.arrayBuffer();
              await ensureAudioContext();
              loadedBuffer = await decodeArrayBuffer(arrBuf);
            } catch (err) {
              console.error(err);
              statusEl.textContent = `Could not decode ${file.name}: ${
                err.message || err
              }`;
              // record failure row and continue
              results.push({
                filename: file.name,
                duration: 0,
                bpm: NaN,
                confidence: 0,
                curve: [],
              });
              renderResultsTable();
              updateProgress(i + 1, files.length);
              continue;
            }

            statusEl.textContent = `Analyzing ${file.name}… (${i + 1}/${
              files.length
            })`;

            // compute options from UI
            const minBPM = Math.max(
              30,
              Math.min(400, Number(minBPMEl.value) || 60)
            );
            const maxBPM = Math.max(
              minBPM + 10,
              Math.min(400, Number(maxBPMEl.value) || 180)
            );
            const frameSize = 1024,
              hopSize = 512;
            $("frameInfo").textContent = `${frameSize} / ${hopSize}`;
            $("rangeValue").textContent = ` ${minBPM}–${maxBPM}`;

            // run main estimate
            const result = estimateBPM(loadedBuffer, {
              frameSize,
              hopSize,
              minBPM,
              maxBPM,
              smoothWin: 3,
            });

            // draw visuals for this file
            drawLine(envCanvas, Array.from(result.envelope || []));
            drawLine(acfCanvas, Array.from(result.acf || []));

            metrics.hidden = false;
            bpmValue.textContent = Number.isFinite(result.bpm)
              ? Math.round(result.bpm) + " BPM"
              : "—";
            bpmNote.textContent = file.name;
            confValue.textContent = fmt(result.confidence, 2);

            lastCurve = [];
            if (perWindowEl.checked) {
              const winSecs = Math.max(2, Number(winSecsEl.value) || 8);
              const hopSecs = Math.max(1, Number(hopSecsEl.value) || 2);
              lastCurve = estimateBPMOverTime(loadedBuffer, {
                winSecs,
                hopSecs,
                frameSize,
                hopSize,
                minBPM,
                maxBPM,
              });
            }
            $("curvePts").textContent = lastCurve.length;
            drawTempoCurve(tempoCanvas, lastCurve, minBPM, maxBPM);

            // Save result row
            results.push({
              filename: file.name,
              duration: loadedBuffer.duration || 0,
              bpm: result.bpm,
              confidence: result.confidence,
              curve: lastCurve,
            });

            renderResultsTable();
            updateProgress(i + 1, files.length);

            // allow a short pause so user can inspect (visual pacing)
            await new Promise((res) => setTimeout(res, 600));
          }

          statusEl.textContent = `✅ Finished analyzing ${files.length} file(s).`;
          analyzeBtn.disabled = false;
          downloadResultsBtn.disabled = results.length > 0 ? false : true;
          exportCsvBtn.disabled = lastCurve.length === 0;
        } else {
          // no files selected, but maybe a single loadedBuffer from mic/demo
          statusEl.textContent = "Analyzing current buffer…";
          try {
            const minBPM = Math.max(
              30,
              Math.min(400, Number(minBPMEl.value) || 60)
            );
            const maxBPM = Math.max(
              minBPM + 10,
              Math.min(400, Number(maxBPMEl.value) || 180)
            );
            const frameSize = 1024,
              hopSize = 512;
            const result = estimateBPM(loadedBuffer, {
              frameSize,
              hopSize,
              minBPM,
              maxBPM,
              smoothWin: 3,
            });

            drawLine(envCanvas, Array.from(result.envelope));
            drawLine(acfCanvas, Array.from(result.acf));
            metrics.hidden = false;
            bpmValue.textContent = Number.isFinite(result.bpm)
              ? Math.round(result.bpm) + " BPM"
              : "—";
            bpmNote.textContent = "Captured / Demo";
            confValue.textContent = fmt(result.confidence, 2);

            lastCurve = [];
            if (perWindowEl.checked) {
              lastCurve = estimateBPMOverTime(loadedBuffer, {
                winSecs: Math.max(2, Number(winSecsEl.value) || 8),
                hopSecs: Math.max(1, Number(hopSecsEl.value) || 2),
                frameSize,
                hopSize,
                minBPM,
                maxBPM,
              });
            }
            $("curvePts").textContent = lastCurve.length;
            drawTempoCurve(tempoCanvas, lastCurve, minBPM, maxBPM);

            results = [
              {
                filename: "Captured/Demo",
                duration: loadedBuffer.duration || 0,
                bpm: result.bpm,
                confidence: result.confidence,
                curve: lastCurve,
              },
            ];
            renderResultsTable();
            updateProgress(1, 1);
            statusEl.textContent = "✅ Done.";
            analyzeBtn.disabled = false;
            downloadResultsBtn.disabled = false;
            exportCsvBtn.disabled = lastCurve.length === 0;
          } catch (err) {
            console.error(err);
            statusEl.textContent = `Analysis error: ${err.message || err}`;
            analyzeBtn.disabled = false;
          }
        }
      });

      // Export tempo curve for the currently displayed lastCurve (existing behavior)
      exportCsvBtn.addEventListener("click", () => {
        if (!lastCurve.length) {
          return;
        }
        const header = "time_sec,bpm,confidence\n";
        const rows = lastCurve
          .map(
            (p) =>
              `${p.time.toFixed(3)},${p.bpm.toFixed(2)},${(p.conf ?? 0).toFixed(
                3
              )}`
          )
          .join("\n");
        const csv = header + rows + "\n";
        const blob = new Blob([csv], { type: "text/csv" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = "tempo_curve.csv";
        document.body.appendChild(a);
        a.click();
        a.remove();
        setTimeout(() => URL.revokeObjectURL(url), 1000);
      });

      // Export one CSV with all results (filename,duration,bpm,confidence)
      downloadResultsBtn.addEventListener("click", () => {
        if (!results.length) return;
        const header = "filename,duration_sec,bpm,confidence\n";
        const rows = results
          .map((r) => {
            const fname = '"' + (r.filename || "").replace(/"/g, '""') + '"';
            const dur = Number.isFinite(r.duration)
              ? r.duration.toFixed(3)
              : "";
            const bpm = Number.isFinite(r.bpm) ? r.bpm.toFixed(2) : "";
            const conf = Number.isFinite(r.confidence)
              ? r.confidence.toFixed(6)
              : "";
            return `${fname},${dur},${bpm},${conf}`;
          })
          .join("\n");
        const csv = header + rows + "\n";
        const blob = new Blob([csv], { type: "text/csv" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = "bpm_results.csv";
        document.body.appendChild(a);
        a.click();
        a.remove();
        setTimeout(() => URL.revokeObjectURL(url), 1000);
      });

      // initial UI state
      updateProgress(0, 0);