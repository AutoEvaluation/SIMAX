"""Automatic quality checks of synthesized audio."""
from __future__ import annotations

import difflib
import json
import os
import re
import time
from pathlib import Path

import numpy as np

from .common import QC_CACHE, chunk_sig, dialogue_spec, rebuild_full, rjson, voice_path, voice_sentences, wjson
from .sustained import runs, sustained_windows

QC_VERSION = 1
WHISPER = os.environ.get("SIMAX_QC_WHISPER", "small.en")
ECAPA = os.environ.get("SIMAX_ECAPA", "speechbrain/spkrec-ecapa-voxceleb")
TOK = re.compile(r"[a-z0-9']+")


def toks(text):
    return [t.strip("'") for t in TOK.findall(text.lower().replace("\u2019", "'")) if t.strip("'")]


class QC:
    def __init__(self, device=None):
        import torch, whisper
        from speechbrain.inference.speaker import EncoderClassifier
        self.torch = torch
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.wm = whisper.load_model(WHISPER, device=self.dev)
        self.enc = EncoderClassifier.from_hparams(source=ECAPA, savedir=str(QC_CACHE / "ecapa"), run_opts={"device": self.dev})
        self.sent = voice_sentences()

    def emb(self, y):
        with self.torch.no_grad():
            e = self.enc.encode_batch(self.torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))[None].to(self.dev)).squeeze().cpu().numpy()
        return e / (np.linalg.norm(e) + 1e-9)

    def ref_emb(self, voice):
        path = QC_CACHE / "refemb" / f"{voice}.npy"
        if path.exists():
            return np.load(path)
        import librosa
        import soundfile as sf

        path.parent.mkdir(parents=True, exist_ok=True)
        y, sr = sf.read(str(voice_path(voice)), dtype="float32")
        y = y.mean(1) if y.ndim > 1 else y
        y = librosa.resample(y, orig_sr=sr, target_sr=16000) if sr != 16000 else y
        emb = self.emb(y)
        np.save(path, emb)
        wjson(path.with_suffix(".f0.json"), {"f0": f0_of(y)})
        return emb

    def check_chunk(self, folder, spec, k, _trimmed=False):
        import soundfile as sf, librosa
        folder = Path(folder); wav = folder / f"chunk_{k:02d}.wav"; sig = chunk_sig(folder, k)
        lo, hi = spec["chunks"][k]; turns = spec["turns"]
        y, sr = sf.read(str(wav), dtype="float32"); y = y.mean(1) if y.ndim > 1 else y
        y16 = librosa.resample(y, orig_sr=sr, target_sr=16000) if sr != 16000 else y; dur = len(y16) / 16000
        wj = folder / f"chunk_{k:02d}.words.json"; words = None
        if wj.exists() and wj.stat().st_mtime >= wav.stat().st_mtime: words = rjson(wj, {}).get("words")
        if words is None:
            r = self.wm.transcribe(y16, language="en", fp16=(self.dev == "cuda"), condition_on_previous_text=False, word_timestamps=True, verbose=None)
            words = [[w["word"], round(w["start"], 2), round(w["end"], 2)] for s in r["segments"] for w in s.get("words", [])]
            wjson(wj, {"words": words, "model": WHISPER})
        hyp, ht = [], []
        for w, a, b in words:
            for t in toks(w): hyp.append(t); ht.append((a, b))
        ref, owner = [], []
        for i in range(lo, hi):
            for t in toks(turns[i]["text"]): ref.append(t); owner.append(i)
        sm = difflib.SequenceMatcher(None, ref, hyp, autojunk=False); mref = [None] * len(ref); mh = set()
        for a, b, n in sm.get_matching_blocks():
            for j in range(n): mref[a + j] = ht[b + j]; mh.add(b + j)
        missing = 1 - sum(x is not None for x in mref) / max(1, len(ref)); added = 1 - len(mh) / max(1, len(hyp))
        vocab = {t for tt in turns for t in toks(tt["text"])}
        leakw = {t for v in (spec["voice_doctor"], spec["voice_patient"]) for t in toks(self.sent.get(v, "")) if len(t) >= 5} - vocab
        leak = sorted({hyp[j] for j in range(len(hyp)) if j not in mh and hyp[j] in leakw})
        if leak and not _trimmed:
            blocks = [(a, b) for a, b, n in sm.get_matching_blocks() if n >= 4]
            lpos = [j for j in range(len(hyp)) if j not in mh and hyp[j] in leakw]
            if blocks and blocks[0][0] <= 3 and max(lpos) < blocks[0][1]:
                cut = max(0.0, ht[blocks[0][1]][0] - 0.25)
                if cut >= 1.0:
                    tmp = wav.with_name(wav.name + ".part.wav"); sf.write(str(tmp), y[int(cut * sr):], sr); tmp.replace(wav)
                    if wj.exists(): wj.unlink()
                    m = self.check_chunk(folder, spec, k, _trimmed=True); m["trimmed_leak_s"] = round(cut, 2)
                    q = rjson(folder / f"chunk_{k:02d}.qc.json"); q["metrics"]["trimmed_leak_s"] = round(cut, 2); wjson(folder / f"chunk_{k:02d}.qc.json", q)
                    return m
        T = {}
        for i, t in zip(owner, mref):
            d = T.setdefault(i, {"n": 0, "f": 0, "s": None, "e": None}); d["n"] += 1
            if t is not None:
                d["f"] += 1; d["s"] = t[0] if d["s"] is None else min(d["s"], t[0]); d["e"] = t[1] if d["e"] is None else max(d["e"], t[1])
        rows, E = [], []
        for i in range(lo, hi):
            d = T.get(i, {"n": 0, "f": 0, "s": None, "e": None}); row = {"turn": i, "role": turns[i]["speaker"], "nw": d["n"], "found": round(d["f"] / max(1, d["n"]), 2)}
            if d["s"] is not None: row.update(s=round(d["s"], 2), e=round(d["e"], 2))
            if d["s"] is not None and d["e"] - d["s"] >= 1.5 and row["found"] >= 0.5:
                seg = y16[int(d["s"] * 16000):int(min(d["e"], d["s"] + 10) * 16000)]
                if len(seg) >= 16000:
                    row["ei"] = len(E); E.append(self.emb(seg)); row["f0"] = f0_of(seg)
            rows.append(row)
        nw = len(ref); long_ = [r for r in rows if r["nw"] >= 8 and r["found"] >= 0.7 and r.get("e") and r["e"] - r["s"] > 0.5]
        art = [r["nw"] / (r["e"] - r["s"]) * 60 for r in long_]
        tone = runs(sustained_windows(y, sr)) if len(y) > 3 * sr else []
        rms = librosa.feature.rms(y=y16, frame_length=1024, hop_length=320)[0]; db = 20 * np.log10(rms + 1e-9); quiet = db < max(-50, np.percentile(db, 95) - 45)
        longest, cur = 0, 0
        for q in quiet: cur = cur + 1 if q else 0; longest = max(longest, cur)
        heads = [r for r in rows if r["nw"] >= 3][:2]; tails = [r for r in rows if r["nw"] >= 3][-2:]
        m = {"dur": round(dur, 2), "words": nw, "missing": round(missing, 3), "added": round(added, 3), "wpm": round(nw / max(dur, 1e-6) * 60, 1),
             "artic_wpm": round(float(np.median(art)), 1) if art else None, "leak": leak, "tone_runs": [[round(a, 1), round(b, 1)] for a, b in tone],
             "longest_silence_s": round(longest * 320 / 16000, 2), "opening_dropped": bool(heads) and all(r["found"] < 0.3 for r in heads),
             "ending_dropped": bool(tails) and all(r["found"] < 0.3 for r in tails), "trim_checked": True}
        np.save(folder / f"chunk_{k:02d}.qcemb.npy", np.array(E) if E else np.zeros((0, 192), dtype=np.float32))
        wjson(folder / f"chunk_{k:02d}.qc.json", {"v": QC_VERSION, "sig": sig, "k": k, "turns_range": [lo, hi], "metrics": m, "turns": rows, "whisper": WHISPER, "at": time.strftime("%F %T")})
        return m

    def check_dialogue(self, folder):
        folder = Path(folder); spec = dialogue_spec(folder); n = 0; trimmed = []
        if spec is None: return 0, None
        for v in (spec["voice_doctor"], spec["voice_patient"]): self.ref_emb(v)
        for k in range(len(spec["chunks"])):
            if chunk_sig(folder, k) is None: continue
            q = rjson(folder / f"chunk_{k:02d}.qc.json")
            if q and q.get("sig") == chunk_sig(folder, k) and q.get("v") == QC_VERSION and q.get("turns_range") == spec["chunks"][k]: continue
            m = self.check_chunk(folder, spec, k); n += 1
            if m.get("trimmed_leak_s"): trimmed.append(k)
        if trimmed:
            rebuild_full(folder)
        v = verdict(folder, spec)
        if v is not None: wjson(folder / "qc_verdict.json", v)
        return n, v


def f0_of(y):
    import librosa
    if len(y) < 4000: return None
    f0 = librosa.yin(y, fmin=65, fmax=420, sr=16000, frame_length=1024, hop_length=320)
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=320)[0][:len(f0)]; f0 = f0[:len(rms)]
    v = f0[(rms >= np.percentile(rms, 50)) & (f0 > 66) & (f0 < 415)]
    return round(float(np.median(v)), 1) if len(v) >= 10 else None


TH = {"miss_fail": 0.20, "miss_susp": 0.10, "add_fail": 0.18, "add_susp": 0.08, "wpm_hi_fail": 225, "wpm_hi_susp": 195, "wpm_lo": 95, "wpm_lo_susp": 110,
      "sil_susp": 6.0, "swap_margin": 0.05, "run_cross": 3, "run_same": 4, "drift_oc": 0.30, "drift_or": 0.25, "drift_oc_hard": 0.15,
      "f0_female_min": 145, "f0_male_max": 175}


def verdict(folder, spec=None):
    T = TH
    folder = Path(folder); spec = spec or dialogue_spec(folder)
    if spec is None: return None
    Q, EM = [], []
    for k in range(len(spec["chunks"])):
        q = rjson(folder / f"chunk_{k:02d}.qc.json"); sig = chunk_sig(folder, k)
        if not q or q.get("sig") != sig or q.get("v") != QC_VERSION: return None
        Q.append(q); EM.append(np.load(folder / f"chunk_{k:02d}.qcemb.npy"))
    ref = {r: np.load(QC_CACHE / "refemb" / f"{spec['voice_' + r]}.npy") for r in ("doctor", "patient")}
    rf0 = {r: (rjson(QC_CACHE / "refemb" / f"{spec['voice_' + r]}.f0.json", {}) or {}).get("f0") for r in ("doctor", "patient")}
    cross = bool(rf0["doctor"] and rf0["patient"] and abs(np.log2(rf0["doctor"] / rf0["patient"])) > 0.5)
    gender = {r: (("male" if rf0[r] < 165 else "female") if rf0[r] else None) for r in ("doctor", "patient")}
    cen = {}
    for role in ("doctor", "patient"):
        V = np.array([EM[k][t["ei"]] for k, q in enumerate(Q) for t in q["turns"] if "ei" in t and t["role"] == role])
        if len(V) < 3: cen[role] = ref[role]; continue
        c = V.mean(0)
        for _ in range(2): keep = V[(V @ c) >= np.percentile(V @ c, 20)]; c = keep.mean(0)
        cen[role] = c / np.linalg.norm(c)
    out = []
    for k, q in enumerate(Q):
        m = q["metrics"]; conf, susp = [], []
        if m["missing"] > T["miss_fail"]: conf.append(f"missing words {m['missing']:.0%}")
        elif m["missing"] > T["miss_susp"]: susp.append(f"missing words {m['missing']:.0%}")
        if m["added"] > T["add_fail"]: conf.append(f"added words {m['added']:.0%}")
        elif m["added"] > T["add_susp"]: susp.append(f"added words {m['added']:.0%}")
        if m["opening_dropped"]: conf.append("opening dropped")
        if m["ending_dropped"]: conf.append("ending dropped")
        if len(m["leak"]) >= 2: conf.append("reference text spoken: " + " ".join(m["leak"][:5]))
        if m["wpm"] > T["wpm_hi_fail"] or m["wpm"] < T["wpm_lo"]: conf.append(f"speaking rate {m['wpm']:.0f} wpm")
        elif m["wpm"] > T["wpm_hi_susp"]: susp.append(f"fast {m['wpm']:.0f} wpm")
        elif m["wpm"] < T["wpm_lo_susp"]: susp.append(f"slow {m['wpm']:.0f} wpm")
        if m["tone_runs"]: conf.append(f"sustained tone {m['tone_runs'][0]}")
        if m["longest_silence_s"] > T["sil_susp"]: susp.append(f"silence {m['longest_silence_s']:.0f}s")
        flags = []
        for t in q["turns"]:
            if "ei" not in t: continue
            e = EM[k][t["ei"]]; x = "patient" if t["role"] == "doctor" else "doctor"
            oc, xc, orr, xr = (float(e @ cen[t["role"]]), float(e @ cen[x]), float(e @ ref[t["role"]]), float(e @ ref[x]))
            f0 = t.get("f0"); g = gender[t["role"]]
            f0_wrong = f0 is not None and g is not None and ((g == "female" and f0 < T["f0_female_min"]) or (g == "male" and f0 > T["f0_male_max"]))
            swap = xc > oc + T["swap_margin"] and xr > orr
            drift = oc < T["drift_oc"] and orr < T["drift_or"] and not swap
            flags.append({"turn": t["turn"], "dur": round(t["e"] - t["s"], 1), "oc": round(oc, 3), "xc": round(xc, 3), "or": round(orr, 3), "xr": round(xr, 3),
                          "f0": f0, "swap": swap, "drift": drift, "f0_wrong": f0_wrong})
        need = T["run_cross"] if cross else T["run_same"]; run, best = [], []
        for f in flags + [None]:
            if f is not None and f["swap"]: run.append(f)
            else:
                if len(run) > len(best): best = run
                run = []
        if len(best) >= need:
            agree = sum(f["f0_wrong"] for f in best) >= len(best) / 2
            (conf if (agree or not cross) else susp).append(f"voice swap turns {best[0]['turn']}-{best[-1]['turn']}")
        elif best and len(best) >= 2: susp.append(f"swap-like turns {best[0]['turn']}-{best[-1]['turn']}")
        dr = [f for f in flags if f["drift"]]
        hard = [f for f in dr if f["dur"] >= 2 and (f["oc"] < T["drift_oc_hard"] or f["f0_wrong"])]
        if hard: conf.append("voice drift turns " + ",".join(str(f["turn"]) for f in hard[:6]))
        elif dr: susp.append("possible drift turns " + ",".join(str(f["turn"]) for f in dr[:6]))
        fw = [f for f in flags if f["f0_wrong"] and not f["swap"] and not f["drift"] and f["dur"] >= 2]
        notes = ["pitch of the other gender turns " + ",".join(str(f["turn"]) for f in fw[:6])] if len(fw) >= 2 else []
        out.append({"k": k, "status": "fail" if conf else "suspect" if susp else "pass", "confirmed": conf, "suspected": susp, "notes": notes, "turns": flags})
    return {"v": QC_VERSION, "sig": [chunk_sig(folder, k) for k in range(len(spec["chunks"]))], "cross": cross, "gender": gender,
            "chunks": out, "at": time.strftime("%F %T")}
