"""MOSS-TTSD v1.0 speech synthesis."""
from __future__ import annotations

import time
from pathlib import Path

from .common import dialogue_spec, rebuild_full, voice_path, voice_sentences, wjson

MODEL = "OpenMOSS-Team/MOSS-TTSD-v1.0"
CODEC = "OpenMOSS-Team/MOSS-Audio-Tokenizer"
DECODING = {"audio_temperature": 1.1, "audio_top_p": 0.9, "audio_top_k": 50, "audio_repetition_penalty": 1.1}
TOKENS_PER_SECOND = 12.5
RATE_OK = (110, 225)


class TTS:
    def __init__(self, low_vram: bool | None = None):
        import soundfile as sf
        import torch
        import torchaudio
        from moss_ttsd_app import _merge_consecutive_speaker_tags, build_conversation, load_backend, normalize_text

        self.torch, self.ta, self.sf = torch, torchaudio, sf
        self.merge, self.build, self.norm = _merge_consecutive_speaker_tags, build_conversation, normalize_text
        torch.backends.cuda.enable_cudnn_sdp(False)
        vram = torch.cuda.get_device_properties(0).total_memory / 2**30
        self.low_vram = vram < 30 if low_vram is None else low_vram
        if self.low_vram:
            from transformers import AutoModel, AutoProcessor

            self.proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True, codec_path=CODEC)
            self.proc.audio_tokenizer.eval()
            self.model = AutoModel.from_pretrained(MODEL, trust_remote_code=True, torch_dtype=torch.bfloat16,
                                                   attn_implementation="sdpa").to("cuda").eval()
            self.device = torch.device("cuda")
            self.sr = int(getattr(self.proc.model_config, "sampling_rate", 24000))
        else:
            self.model, self.proc, self.device, self.sr = load_backend(MODEL, CODEC, "cuda", "auto")
        self.sent = voice_sentences()

    def decode(self, out):
        if not self.low_vram:
            return self.proc.decode(out)
        self.model.to("cpu")
        self.torch.cuda.empty_cache()
        self.proc.audio_tokenizer.to("cuda")
        try:
            return self.proc.decode(out)
        finally:
            self.proc.audio_tokenizer.to("cpu")
            self.torch.cuda.empty_cache()
            self.model.to("cuda")

    def ref_wav(self, voice: str):
        wav, rate = self.sf.read(str(voice_path(voice)), dtype="float32", always_2d=True)
        wav = self.torch.from_numpy(wav.T.copy()).mean(0, keepdim=True)
        return self.ta.functional.resample(wav, rate, self.sr)

    def generate(self, text, ref_codes, prompt_audio, words: int, attempts: int = 3):
        conv, mode, _ = self.build(text, ref_codes, prompt_audio, self.proc)
        batch = self.proc(conv, mode=mode)
        max_new = int((words / 2.5) * TOKENS_PER_SECOND * 1.8) + 300
        best = None
        for attempt in range(attempts):
            with self.torch.no_grad():
                out = self.model.generate(input_ids=batch["input_ids"].to(self.device),
                                          attention_mask=batch["attention_mask"].to(self.device),
                                          max_new_tokens=max_new, **DECODING)
            audio = self.decode(out)[0].audio_codes_list[0].detach().float().cpu().numpy().reshape(-1)
            wpm = words / max(len(audio) / self.sr, 1e-6) * 60
            capped = len(audio) / self.sr >= (max_new - 300) / TOKENS_PER_SECOND * 0.98
            score = abs(wpm - 160) + (1000 if capped else 0)
            if best is None or score < best[0]:
                best = (score, audio, attempt + 1, round(wpm, 1))
            if RATE_OK[0] <= wpm <= RATE_OK[1] and not capped:
                break
        return best

    def fill_folder(self, folder) -> list[int]:
        folder = Path(folder)
        spec = dialogue_spec(folder)
        turns, doctor, patient = spec["turns"], spec["voice_doctor"], spec["voice_patient"]
        refs = [self.ref_wav(doctor), self.ref_wav(patient)]
        prefix = f"[S1] {self.sent[doctor]}[S2] {self.sent[patient]}"
        prefix_patient_first = f"[S2] {self.sent[patient]}[S1] {self.sent[doctor]}"
        ref_codes = self.proc.encode_audios_from_wav(refs, sampling_rate=self.sr)
        prompt = self.proc.encode_audios_from_wav([self.torch.cat(refs, dim=-1)], sampling_rate=self.sr)[0]
        prompt_patient_first = self.proc.encode_audios_from_wav([self.torch.cat(refs[::-1], dim=-1)],
                                                                sampling_rate=self.sr)[0]
        made = []
        for k, (a, b) in enumerate(spec["chunks"]):
            path = folder / f"chunk_{k:02d}.wav"
            if path.exists():
                continue
            words = sum(len(t["text"].split()) for t in turns[a:b])
            body = "".join(f"[S{1 if t['speaker'] == 'doctor' else 2}] {t['text']} " for t in turns[a:b])
            patient_first = turns[a]["speaker"] == "patient"
            text = self.norm(self.merge((prefix_patient_first if patient_first else prefix) + body))
            started = time.time()
            best = self.generate(text, ref_codes, prompt_patient_first if patient_first else prompt, words)
            tmp = path.with_name(path.name + ".part.wav")
            self.sf.write(str(tmp), best[1], self.sr)
            tmp.replace(path)
            wjson(folder / f"chunk_{k:02d}.attempts.json",
                  {"attempts_used": best[2], "kept_wpm": best[3], "seconds": round(time.time() - started, 1)})
            made.append(k)
        rebuild_full(folder)
        return made
