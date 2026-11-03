// 브라우저 블러(10/31, H-44·H-45): 원본은 사용자 기기에 두고, 고른 구간을 2fps로 읽어 얼굴을 2배 검출로 흐린 뒤
// 1분 조각 mp4로 만들어 서버에 올린다. 서버는 조각마다 Meta에 묻는다. 원본과 흐리지 않은 프레임은 기기 밖으로 나가지 않는다.
// 아이폰 11 Pro에서 9조각 2배 장당 1.15초, 맥 크롬은 장당 0.3초 안팎(H-45). 처음 몇 장이 느린 기기는 원본을 올려 서버가 흐린다.
(() => {
  const LIBS = [
    "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.wasm.min.js",
    "https://cdn.jsdelivr.net/npm/mp4box@0.5.2/dist/mp4box.all.min.js",
    "https://cdn.jsdelivr.net/npm/mp4-muxer@5.2.1/build/mp4-muxer.min.js",
  ];
  // 외부 스크립트 무결성(SRI, 10/31 보안 검토): 배포처의 파일이 바뀌면 받지 않는다. 버전을 올리면 다시 계산한다
  // (curl -s URL | openssl dgst -sha384 -binary | openssl base64 -A). onnxruntime이 뒤에 받는 .wasm은 여기서 확인하지 못한다
  const INTEGRITY = {
    [LIBS[0]]: "sha384-6hgW9I+eQKl5CrHLstudPd4L3E2eTYHm+qNKvf4pI+RZnfBJILOHAiaalHV+trgG",
    [LIBS[1]]: "sha384-9oK16t6TCKqCB5OpB/LwRkNa48prlRFQPTYw8tb9uj699JvQ1wjxIq7YO8Z4hn3a",
    [LIBS[2]]: "sha384-53pBfagDTItQJ22xfdceasmWxlheAxIBk28yeg05+R9oWPW02xExIznXrB0AFPe4",
  };
  const MODEL = "/static/models/yunet_dyn.onnx";  // tapme/browser_model.py로 만든다
  const FPS = 2, CHUNK_SEC = 60, STEP_SEC = 50;  // 서버 tapme/scan.py와 같게
  const THR = 0.5, NMS_IOU = 0.3, TILE_OVERLAP = 48;
  const PROBE_FRAMES = 10, MAX_SEC_PER_FRAME = 1.0;  // H-45 판정 기준: 2배 장당 1초를 넘는 기기는 서버로
  const BITRATE = 2_000_000;  // 2fps 1080p 흐림본 1분 약 15MB
  const MAX_RESTARTS = 8;  // 처리 도중 디코더가 멈췄을 때 새로 만들어 이어 가는 횟수
  // 폰은 9조각(4조각은 아이폰 사파리가 메모리 부족으로 닫힘, H-45), PC는 4조각
  // 9조각은 아이폰 사파리 메모리 때문(4조각이면 탭이 닫혔다, H-45). 안드로이드 크롬·PC는 4조각(계산이 20~30% 적다)
  const IOS = /iPhone|iPad/i.test(navigator.userAgent) || (/Macintosh/.test(navigator.userAgent) && navigator.maxTouchPoints > 1);
  // 그래픽칩(WebGPU)은 쓰지 않는다: 아이폰은 첫 장 뒤 멈췄고(H-44), 갤럭시 S20+는 더 느리고(장당 1.51초 vs 1.34초)
  // 얼굴을 36% 덜 찾았다(2,254 vs 3,532개, 정밀도 차이, 10/31) — 블러 품질이 떨어진다

  class TooSlow extends Error {}

  // mp4 트랙 행렬(16.16 고정소수 a,b,u,c,d,v,…)이 기본 방향인지: a=d=1.0, b=c=0. 없으면 기본으로 본다
  function isUpright(m) {
    return !m || m.length < 5 || (m[0] === 65536 && m[1] === 0 && m[3] === 0 && m[4] === 65536);
  }

  function supported() {
    return typeof VideoDecoder === "function" && typeof VideoEncoder === "function" && typeof VideoFrame === "function"
      && typeof OffscreenCanvas === "function" && typeof WebAssembly === "object";
  }

  function loadScript(src) {
    return new Promise((ok, bad) => {
      if (document.querySelector(`script[src="${src}"]`)) return ok();
      const s = document.createElement("script");
      s.src = src; s.crossOrigin = "anonymous"; if (INTEGRITY[src]) s.integrity = INTEGRITY[src];
      s.onload = ok; s.onerror = () => bad(new Error("처리 도구를 받지 못했어요"));
      document.head.append(s);
    });
  }

  function chunkPlan(offset, length) {
    const starts = [0];
    while (starts[starts.length - 1] + CHUNK_SEC < length) starts.push(starts[starts.length - 1] + STEP_SEC);
    return starts.map((s) => [offset + s, Math.min(CHUNK_SEC, length - s)]);
  }

  // ---- mp4 읽기: 목차(moov)를 찾고, 구간 앞 키프레임부터 조금씩 읽어 하드웨어 디코더로 푼다 ----
  class Mp4Source {
    constructor(file) { this.file = file; }

    async append(at) {
      const buf = await this.file.slice(at, at + (8 << 20)).arrayBuffer();
      buf.fileStart = at;
      return [this.mp4.appendBuffer(buf), buf.byteLength];
    }

    async open() {
      const mp4 = this.mp4 = MP4Box.createFile();
      let err = null;
      mp4.onError = (e) => { err = e; };
      mp4.onReady = (info) => { this.info = info; };
      let pos = 0;
      while (!this.info && !err && pos < this.file.size) {  // moov가 끝에 있으면 mp4box가 끝으로 건너뛰게 한다
        const [next, n] = await this.append(pos);
        pos = Number.isFinite(next) && next !== pos ? next : pos + n;
      }
      const track = this.info && this.info.videoTracks[0];
      if (err || !track) throw new Error("영상을 풀 수 없어요");
      // 세워 찍은 영상은 프레임이 누운 채 회전 표시만 있다. 그대로 흐리면 누운 조각이 Meta로 가 서버(ffmpeg가 바로 세움)로 넘긴다
      // 기본 방향(행렬 a=d=1.0, b=c=0)이 아니면 모두 서버로. 90°·270°만 걸러 180°(폰을 반대로 들고 찍음)·좌우 반전은
      // 거꾸로 된 프레임을 흐려 그대로 Meta로 보냈다 — 거꾸로 된 얼굴은 놓칠 수 있다(11/3 점검)
      const m = track.matrix || [];
      if (!isUpright(m)) throw new Error("회전된 영상은 서버에서 처리해요");
      this.track = track;
      let description;
      for (const entry of mp4.getTrackById(track.id).mdia.minf.stbl.stsd.entries) {
        const box = entry.avcC || entry.hvcC || entry.vpcC || entry.av1C;
        if (box) { const ds = new DataStream(undefined, 0, DataStream.BIG_ENDIAN); box.write(ds); description = new Uint8Array(ds.buffer, 8); break; }
      }
      this.width = track.video.width; this.height = track.video.height;
      // 사파리 디코더는 까다로워 첫 장을 못 풀면 설정을 바꿔 본다(H-45 시험 페이지)
      this.configs = [
        { codec: track.codec, description, codedWidth: this.width, codedHeight: this.height },
        { codec: track.codec, description, codedWidth: this.width, codedHeight: this.height, useDts: true },
        { codec: track.codec, description, useDts: true },
        { codec: track.codec, description, hardwareAcceleration: "prefer-software", useDts: true },
      ];
    }

    // [from, to) 구간에서 0.5초마다 한 장씩 onFrame(t, frame)을 부른다. frame은 부른 뒤 여기서 닫는다
    // 아이폰 사파리 디코더는 처리 도중 "Decoder failure"로 멈췄다(5분 영상의 100~200장째, 10/31). 그러면 새 디코더로
    // 마지막으로 처리한 장의 다음부터 다시 푼다(최대 MAX_RESTARTS번). 첫 장도 못 풀면 설정을 바꿔 본다
    async frames(from, to, onFrame) {
      const errors = [], state = { done: 0, last: null, own: null };
      // onFrame이 던진 것(속도 재기 멈춤·업로드 실패)은 디코더 고장이 아니라 그대로 올린다. 고장으로 보고 다시 풀면
      // 속도 재기가 10장을 넘겨 "12/10장"이 됐다(10/31 아이폰)
      const counted = async (t, f) => {
        try { await onFrame(t, f); } catch (e) { state.own = e; throw e; }
        state.last = t;
      };
      let cur = from, restarts = 0;
      for (let k = 0; k < this.configs.length;) {
        const { useDts, ...config } = this.configs[k];
        let ok = false;
        try { ok = (await VideoDecoder.isConfigSupported(config)).supported; } catch { ok = false; }
        if (!ok) { k++; continue; }
        try { return await this.decode(cur, to, counted, config, useDts, state); } catch (e) {
          if (e instanceof TooSlow || e === state.own) throw e;
          errors.push(String(e.message || e));
          if (state.done === 0) { k++; continue; }  // 첫 장도 못 풀었다: 다른 설정으로
          if (++restarts > MAX_RESTARTS) throw e;
          cur = state.last + 1 / FPS;  // 같은 설정, 새 디코더로 다음 장부터
        }
      }
      throw new Error("이 기기에서는 영상을 풀 수 없어요" + (errors.length ? ` (${errors[0]})` : ""));
    }

    async decode(from, to, onFrame, config, useDts, state) {
      const mp4 = this.mp4, id = this.track.id, samples = [];
      mp4.onSamples = (tid, user, ss) => { samples.push(...ss); };
      mp4.setExtractionOptions(id, null, { nbSamples: 60 });
      const seek = mp4.seek(from, true);  // 구간 앞 키프레임부터
      mp4.start();
      let pos = seek.offset, i = 0, next = from, wake = null, failed = null;
      const pending = [];
      const ping = () => { if (wake) { wake(); wake = null; } };
      const dec = new VideoDecoder({
        output: (f) => {
          const t = f.timestamp / 1e6;
          if (next < to && t + 0.001 >= next) { pending.push(f); next += 1 / FPS; } else f.close();
          ping();
        },
        error: (e) => { failed = e; ping(); },
      });
      dec.configure(config);
      dec.ondequeue = ping;
      let flushed = false;
      try {
        while (next < to || pending.length) {
          if (failed) throw failed;
          // 디코더에 넣어 두는 조각은 세 개, 기다리는 프레임은 한 장까지(폰 메모리)
          while (!failed && pending.length < 1 && dec.decodeQueueSize < 3 && i < samples.length) {
            const s = samples[i++];
            const ts = (useDts ? s.dts : s.cts) / s.timescale;
            if (ts > to + 2) { i = samples.length; break; }
            dec.decode(new EncodedVideoChunk({ type: s.is_sync ? "key" : "delta", timestamp: 1e6 * ts,
                                               duration: 1e6 * s.duration / s.timescale, data: s.data }));
            mp4.releaseUsedSamples(id, s.number + 1);
          }
          if (pending.length) {
            const f = pending.shift();
            try { await onFrame(f.timestamp / 1e6, f); state.done++; } finally { f.close(); }
            continue;
          }
          if (i >= samples.length) {
            const last = samples.length ? (useDts ? samples[samples.length - 1].dts : samples[samples.length - 1].cts) / samples[samples.length - 1].timescale : from;
            if (pos < this.file.size && last <= to + 2) { pos += (await this.append(pos))[1]; continue; }
            if (flushed) break;
            flushed = true; await dec.flush(); continue;
          }
          await new Promise((ok) => { wake = ok; setTimeout(ok, 20); });
        }
      } finally {
        for (const f of pending) f.close();
        if (dec.state !== "closed") dec.close();
        mp4.stop();
        mp4.unsetExtractionOptions(id);  // 다시 시작할 때 같은 트랙을 두 번 꺼내지 않게
      }
    }
  }

  // ---- 얼굴 찾기·흐리기: 서버(tapme/anonymize.py)와 같은 YuNet 해석, 2배는 조각으로 나눠(메모리) ----
  class Blurrer {
    async init(W, H) {
      ort.env.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency || 2);
      this.ep = "wasm";
      this.session = await ort.InferenceSession.create(MODEL, { executionProviders: ["wasm"] });
      const grid = IOS ? 3 : 2, cw = Math.ceil(W / grid), ch = Math.ceil(H / grid), O = TILE_OVERLAP;
      this.tiles = [];
      for (let r = 0; r < grid; r++) for (let c = 0; c < grid; c++) {
        const x = Math.max(0, c * cw - (c ? O : 0)), y = Math.max(0, r * ch - (r ? O : 0));
        this.tiles.push([x, y, Math.min(cw + O * (c ? 1 : 0) + O, W - x), Math.min(ch + O * (r ? 1 : 0) + O, H - y)]);
      }
      this.pw = Math.ceil(Math.max(...this.tiles.map((t) => t[2])) * 2 / 32) * 32;
      this.ph = Math.ceil(Math.max(...this.tiles.map((t) => t[3])) * 2 / 32) * 32;
      this.work = new OffscreenCanvas(this.pw, this.ph).getContext("2d", { willReadFrequently: true });
      this.out = new OffscreenCanvas(W, H);
      this.octx = this.out.getContext("2d");
      this.buf = new Float32Array(3 * this.pw * this.ph);
      this.W = W; this.H = H;
    }

    async detect() {
      const { pw, ph } = this, n = pw * ph, rgba = this.work.getImageData(0, 0, pw, ph).data, data = this.buf;
      for (let i = 0; i < n; i++) { data[i] = rgba[4 * i + 2]; data[n + i] = rgba[4 * i + 1]; data[2 * n + i] = rgba[4 * i]; }  // BGR 0~255
      const out = await this.session.run({ [this.session.inputNames[0]]: new ort.Tensor("float32", data, [1, 3, ph, pw]) });
      const boxes = [], scores = [];
      for (const s of [8, 16, 32]) {
        const cols = pw / s, cls = out[`cls_${s}`].data, obj = out[`obj_${s}`].data, bb = out[`bbox_${s}`].data;
        for (let i = 0; i < cls.length; i++) {
          const sc = Math.sqrt(Math.min(Math.max(cls[i], 0), 1) * Math.min(Math.max(obj[i], 0), 1));
          if (sc < THR) continue;
          const r = Math.floor(i / cols), c = i % cols;
          const cx = (c + bb[4 * i]) * s, cy = (r + bb[4 * i + 1]) * s, w = Math.exp(bb[4 * i + 2]) * s, h = Math.exp(bb[4 * i + 3]) * s;
          boxes.push([cx - w / 2, cy - h / 2, w, h]); scores.push(sc);
        }
      }
      return nms(boxes, scores);
    }

    // 프레임을 흐린 전체 화면(this.out)을 돌려준다. 사파리는 VideoFrame에서 일부를 잘라 그리면 자를 위치를 무시해
    // (H-45) 먼저 전체를 그린 뒤 거기서 조각을 자른다
    async blur(frame) {
      const { W, H, octx } = this;
      octx.drawImage(frame, 0, 0, W, H);
      let found = [];
      for (const [sx, sy, sw, sh] of this.tiles) {
        this.work.clearRect(0, 0, this.pw, this.ph);
        this.work.drawImage(this.out, sx, sy, sw, sh, 0, 0, sw * 2, sh * 2);
        found.push(...(await this.detect()).map(([x, y, w, h]) => [sx + x / 2, sy + y / 2, w / 2, h / 2]));
      }
      found = nms(found, found.map(([, , w, h]) => w * h));  // 조각이 겹친 곳에서 두 번 찾은 얼굴
      pixelate(octx, found);
      return found.length;
    }
  }

  function nms(boxes, scores) {
    const order = scores.map((s, i) => i).sort((a, b) => scores[b] - scores[a]), keep = [];
    const iou = (a, b) => {
      const x1 = Math.max(a[0], b[0]), y1 = Math.max(a[1], b[1]);
      const x2 = Math.min(a[0] + a[2], b[0] + b[2]), y2 = Math.min(a[1] + a[3], b[1] + b[3]);
      const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
      return inter / (a[2] * a[3] + b[2] * b[3] - inter);
    };
    for (const i of order) if (keep.every((k) => iou(boxes[i], boxes[k]) <= NMS_IOU)) keep.push(i);
    return keep.map((i) => boxes[i]);
  }

  // 서버 anonymize.pixelate_boxes와 같게: 사방 25% 넓혀 굵은 모자이크(블록 = 폭/12, 최소 8px)
  function pixelate(ctx, boxes) {
    for (const [x0, y0, w0, h0] of boxes) {
      const x = Math.max(0, Math.floor(x0 - w0 * 0.25)), y = Math.max(0, Math.floor(y0 - h0 * 0.25));
      const w = Math.ceil(w0 * 1.5), h = Math.ceil(h0 * 1.5);
      const block = Math.max(8, Math.floor(w / 12)), sw = Math.max(1, Math.floor(w / block)), sh = Math.max(1, Math.floor(h / block));
      const tmp = new OffscreenCanvas(sw, sh), tctx = tmp.getContext("2d");
      tctx.drawImage(ctx.canvas, x, y, w, h, 0, 0, sw, sh);
      ctx.imageSmoothingEnabled = false; ctx.drawImage(tmp, 0, 0, sw, sh, x, y, w, h); ctx.imageSmoothingEnabled = true;
    }
  }

  // ---- 1분 조각 mp4 만들기(WebCodecs H.264 + mp4-muxer) ----
  async function encoderConfig(W, H, framerate = FPS, bitrate = BITRATE) {
    for (const codec of ["avc1.42E02A", "avc1.4D002A", "avc1.640028"]) {
      const c = { codec, width: W, height: H, bitrate, framerate, avc: { format: "avc" } };
      try { if ((await VideoEncoder.isConfigSupported(c)).supported) return c; } catch { /* 다음 후보 */ }
    }
    throw new Error("이 기기에서는 영상을 만들 수 없어요");
  }

  class ChunkWriter {
    constructor(index, start, seconds, config) {
      this.index = index; this.start = start; this.end = start + seconds; this.n = 0; this.error = null;
      this.target = new Mp4Muxer.ArrayBufferTarget();
      this.muxer = new Mp4Muxer.Muxer({ target: this.target, video: { codec: "avc", width: config.width, height: config.height },
                                        fastStart: "in-memory", firstTimestampBehavior: "offset" });
      this.enc = new VideoEncoder({ output: (chunk, meta) => this.muxer.addVideoChunk(chunk, meta), error: (e) => { this.error = e; } });
      this.enc.configure(config);
    }

    add(canvas, t) {
      if (this.error) throw this.error;
      const vf = new VideoFrame(canvas, { timestamp: Math.round((t - this.start) * 1e6), duration: 1e6 / FPS });
      this.enc.encode(vf, { keyFrame: this.n % 10 === 0 });
      vf.close();
      this.n++;
    }

    async finish() {
      await this.enc.flush();
      this.enc.close();
      if (this.error) throw this.error;
      this.muxer.finalize();
      return new Blob([this.target.buffer], { type: "video/mp4" });
    }
  }

  // 전체 흐름. onProgress({done, total, secLeft}), createJob(plan) → 서버가 만든 조각 계획(처음 몇 장으로 이 기기가 충분히
  // 빠를 때만 부른다), uploadChunk(i, blob). force: 속도와 상관없이 기기에서(시험용, 주소 ?local=force). 느린 기기면 "slow"를 돌려준다(작업은 만들지 않음 — 원본을 올려 서버가 흐린다)
  // skip: 서버가 이미 받은 조각 번호(이어 처리하기). 주면 작업을 새로 만들지 않고, 빠진 첫 조각의 시작부터 다시 처리한다
  async function run({ file, start, end, onProgress, createJob, uploadChunk, force = false, skip = null }) {
    for (const src of LIBS) await loadScript(src);
    const source = new Mp4Source(file);
    await source.open();
    const blurrer = new Blurrer();
    await blurrer.init(source.width, source.height);
    const config = await encoderConfig(source.width, source.height);
    const plan = chunkPlan(start, end - start);
    let from = start;
    if (skip) {
      const first = plan.findIndex((_, i) => !skip.has(i));
      if (first < 0) return "done";
      from = plan[first][0];
    }
    const total = Math.ceil((end - from) * FPS);
    const writers = new Map();
    let frames = 0, created = Boolean(skip), tProbe = 0, uploadError = null, uploads = Promise.resolve();
    if (skip) force = true;
    const queueUpload = (i, blob) => {
      uploads = uploads.then(async () => {
        for (let k = 0; ; k++) {
          try { return await uploadChunk(i, blob); } catch (e) { if (k >= 2) throw e; await new Promise((ok) => setTimeout(ok, 3000)); }
        }
      }).catch((e) => { uploadError = e; });
    };
    const create = async () => {
      created = true;
      const serverPlan = await createJob(plan);
      if (serverPlan.length !== plan.length || serverPlan.some(([s], i) => Math.abs(s - plan[i][0]) > 0.01)) throw new Error("조각 계획이 서버와 달라요");
    };
    try {
      await source.frames(from, end, async (t, frame) => {
        if (uploadError) throw uploadError;
        await blurrer.blur(frame);
        frames++;
        if (frames === 2) tProbe = performance.now();  // 첫 장은 준비(모델·디코더 데우기)라 빼고 잰다
        plan.forEach(([s0, sec], i) => {
          if (skip && skip.has(i)) return;  // 이미 서버에 있는 조각
          if (t >= s0 - 1e-6 && t < s0 + sec) {
            if (!writers.has(i)) writers.set(i, new ChunkWriter(i, s0, sec, config));
            writers.get(i).add(blurrer.out, t);
          }
        });
        for (const [i, w] of writers) {  // 다음 장이 이 조각 밖이면 닫아 올린다
          if (t + 1 / FPS >= w.end - 1e-6) { writers.delete(i); queueUpload(i, await w.finish()); }
        }
        const per = frames > 2 ? (performance.now() - tProbe) / 1000 / (frames - 2) : 0;
        if (!created && frames >= PROBE_FRAMES) {
          if (per > MAX_SEC_PER_FRAME && !force) throw new TooSlow(`장당 ${per.toFixed(2)}초`);
          await create();
        }
        if (onProgress) onProgress({ done: frames, total, secLeft: Math.max(0, (total - frames) * per), perFrame: per });
      });
      if (!created) await create();  // 10장보다 짧은 구간
      for (const [i, w] of writers) queueUpload(i, await w.finish());
      writers.clear();
      await uploads;
      if (uploadError) throw uploadError;
      return "done";
    } catch (e) {
      for (const w of writers.values()) { try { w.enc.close(); } catch { /* 이미 닫힘 */ } }
      if (e instanceof TooSlow) return "slow";
      throw e;
    }
  }

  // 2단계: 고른 랠리 구간을 원본에서 다시 압축하지 않고 잘라 낸다(mp4box 분할: 영상·소리 따로, 화질·소리 그대로).
  // 구간 앞 키프레임부터라 요청보다 조금 넓다. start = 첫 영상 프레임(키프레임)의 원본 시각 — 서버가 시각을 맞추는 기준
  async function extractSegment(file, from, to) {
    await loadScript(LIBS[1]);
    const mp4 = MP4Box.createFile();
    let info = null, err = null;
    mp4.onError = (e) => { err = e; };
    mp4.onReady = (i) => { info = i; };
    const append = async (at) => {
      const buf = await file.slice(at, at + (8 << 20)).arrayBuffer();
      buf.fileStart = at;
      return [mp4.appendBuffer(buf), buf.byteLength];
    };
    let pos = 0;
    while (!info && !err && pos < file.size) { const [next, n] = await append(pos); pos = Number.isFinite(next) && next !== pos ? next : pos + n; }
    if (err || !info || !info.videoTracks[0]) throw new Error("영상을 풀 수 없어요");
    const tracks = [info.videoTracks[0], info.audioTracks[0]].filter(Boolean), vid = tracks[0].id;
    const segs = Object.fromEntries(tracks.map((t) => [t.id, []]));
    for (const t of tracks) mp4.setSegmentOptions(t.id, null, { nbSamples: 10 });  // 끝에서 덜 나온 조각이 짧게
    mp4.onSegment = (id, user, buffer) => { segs[id].push(buffer); };
    const init = mp4.initializeSegmentation();
    const trak = mp4.getTrackById(vid);
    let key = 0;  // from 이전 마지막 키프레임(mp4box seek가 고르는 곳)
    trak.samples.forEach((s, k) => { if (s.is_sync && s.cts / s.timescale <= from) key = k; });
    const start = trak.samples[key].cts / trak.samples[key].timescale;
    pos = mp4.seek(from, true).offset;
    mp4.start();
    // 소리는 파일 안에서 영상과 섞여 있어 영상만 끝까지 받고 멈추면 소리가 짧았다(16초, 10/31) → 끝 뒤로 3초 더 읽는다
    const past = (id) => { const t = mp4.getTrackById(id), s = t.samples[t.nextSample - 1]; return s && s.cts / s.timescale > to + 3; };
    while (pos < file.size && !tracks.every((t) => past(t.id))) pos += (await append(pos))[1];
    mp4.flush();
    const blob = (id) => new Blob([init.find((x) => x.id === id).buffer, ...segs[id]], { type: "video/mp4" });
    return { video: blob(vid), audio: tracks[1] ? blob(tracks[1].id) : null, start };
  }

  // 시작 전 속도 재기: 구간 앞 몇 장을 실제처럼 풀고 흐려 장당 시간을 잰다(올리지 않음). 첫 장(준비)은 빼고 잰다
  // onStep(단계, 지금까지 장 수, 잴 장 수): 화면이 어디까지 됐는지 보여 준다(폰에서 멈춘 것처럼 보였다, 10/31)
  async function probe(file, start, onStep = () => {}) {
    onStep("libs", 0, PROBE_FRAMES);
    for (const src of LIBS) await loadScript(src);
    onStep("open", 0, PROBE_FRAMES);
    const source = new Mp4Source(file);
    await source.open();
    onStep("model", 0, PROBE_FRAMES);
    const blurrer = new Blurrer();
    await blurrer.init(source.width, source.height);
    let n = 0, t1 = 0;
    const stop = new Error("stop");
    try {
      await source.frames(start, start + PROBE_FRAMES / FPS + 1, async (t, f) => {
        await blurrer.blur(f);
        if (++n === 2) t1 = performance.now();
        onStep("measure", n, PROBE_FRAMES);
        if (n >= PROBE_FRAMES) throw stop;
      });
    } catch (e) { if (e !== stop) throw e; }
    return { per: n > 2 ? (performance.now() - t1) / 1000 / (n - 2) : Infinity, ep: blurrer.ep, tiles: blurrer.tiles.length };
  }

  // 미리 재기(10/31 사용자: 영상을 올리기 전에 이 기기가 어느 정도인지 알고 싶다): 사람이 없는 1080p 30fps 시험 영상을
  // 이 기기에서 만들어(밖으로 보내지 않음) probe와 똑같이 잰다. 계산은 화소 수에 비례해(조각 수는 같고 조각 크기가 커진다)
  // 다른 해상도는 화소 수 비율로 바꿔 본다. 얼굴이 없어도 검출 시간은 거의 같다(모델 계산이 대부분)
  const BENCH_W = 1920, BENCH_H = 1080, BENCH_FPS = 30;
  async function benchFile() {
    await loadScript(LIBS[2]);
    const config = await encoderConfig(BENCH_W, BENCH_H, BENCH_FPS, 8_000_000);
    const target = new Mp4Muxer.ArrayBufferTarget();
    const muxer = new Mp4Muxer.Muxer({ target, video: { codec: "avc", width: BENCH_W, height: BENCH_H }, fastStart: "in-memory" });
    let error = null;
    const enc = new VideoEncoder({ output: (chunk, meta) => muxer.addVideoChunk(chunk, meta), error: (e) => { error = e; } });
    enc.configure(config);
    const canvas = new OffscreenCanvas(BENCH_W, BENCH_H), ctx = canvas.getContext("2d");
    const n = Math.ceil((PROBE_FRAMES / FPS + 1.5) * BENCH_FPS);  // probe가 읽는 길이 + 여유
    for (let i = 0; i < n; i++) {
      // 체육관처럼 벽·바닥과 움직이는 공(사람 없음). 매 장면이 달라야 압축·풀기가 실제와 비슷하다
      ctx.fillStyle = "#e8e4dc"; ctx.fillRect(0, 0, BENCH_W, BENCH_H);
      ctx.fillStyle = "#c9a26b"; ctx.fillRect(0, BENCH_H * 0.45, BENCH_W, BENCH_H * 0.55);
      for (let k = 0; k < 24; k++) {
        ctx.fillStyle = `hsl(${k * 15} 60% 50%)`; ctx.beginPath();
        ctx.arc((k * 83 + i * 9 * (1 + k % 3)) % BENCH_W, 200 + (k % 6) * 140 + 50 * Math.sin(i / 8 + k), 30 + k % 4 * 10, 0, 7); ctx.fill();
      }
      const vf = new VideoFrame(canvas, { timestamp: Math.round(i * 1e6 / BENCH_FPS), duration: Math.round(1e6 / BENCH_FPS) });
      enc.encode(vf, { keyFrame: i % BENCH_FPS === 0 });
      vf.close();
      while (enc.encodeQueueSize > 4 && !error) await new Promise((ok) => setTimeout(ok, 5));
      if (error) throw error;
    }
    await enc.flush();
    enc.close();
    if (error) throw error;
    muxer.finalize();
    return new File([target.buffer], "bench.mp4", { type: "video/mp4" });
  }

  // onStep: probe와 같은 단계 + 맨 앞 "make"(시험 영상 만들기). 돌려주는 per는 1080p 기준 장당 초
  async function bench(onStep = () => {}) {
    onStep("make", 0, PROBE_FRAMES);
    const file = await benchFile();
    return { ...(await probe(file, 0, onStep)), px: BENCH_W * BENCH_H };
  }

  window.BrowserBlur = { supported, isUpright, chunkPlan, run, probe, bench, extractSegment, FPS, MAX_SEC_PER_FRAME, PROBE_FRAMES };
})();
