# SIMAX

SIMAX is a dataset and pipeline for simulated clinician-patient visits in which the communication behaviors are controlled in advance.
It is built to test communication coding systems against known targets.

- **705 full-visit scripts** across orthopedics, obstetrics and rheumatology
- **1,410 audio records**: every script is synthesized with two clinician voice sources, as dry and wet audio
- **21 target behaviors** from five codebooks: Global, WISER, Intervention (5A's), Patient Behavior, and SDOH & Weight

The dataset is on Hugging Face: [Nomothetic/SIMAX](https://huggingface.co/datasets/Nomothetic/SIMAX).

SIMAX is a research resource. Its dialogues are simulated and must not be used for diagnosis or patient care.

## Quick start

```bash
pip install -e .
python -m unittest discover -s tests
```

To generate one dialogue, set `OPENAI_API_KEY` in `.env` and fill in the two profile templates (see [Required setup](#required-setup)):

```bash
pip install -e ".[generation]"
cp .env.example .env
set -a; source .env; set +a
python scripts/generate_dialogues.py --out-dir runs/dialogues --limit 1
```

## Repository layout

| Path | Contents |
|---|---|
| `configs/` | Generation settings, persona plan, condition bank, label definitions, voice list |
| `src/simax/` | Prompts and generation, speech synthesis, automatic checks |
| `scripts/` | Pipeline and evaluation entry points |
| `codebooks/` | Codebooks |
| `resources/audio_assets/` | Room impulse response and background sounds |

## License

The code is released under the [MIT License](LICENSE). The sound files in `resources/audio_assets/` are third-party recordings and keep their original licenses.

## Required setup

These inputs are not included in the repository.

| Input | Where to put it | Used by |
|---|---|---|
| Visit profiles | Replace every `null` in `configs/dialogue_profile.json` and `configs/topic_profile.json` with your own values for visit length, conversational shape, small talk and visit mix. Generation stops with an error until both are filled in. | `scripts/generate_dialogues.py`, `scripts/spoken_rewrite.py` |
| Reference voices | `$SIMAX_VOICE_DIR/wav16k/<wav>`. `configs/voices.json` lists each clip with its source and spoken sentence. The Internal-Extended clinician voice (`ruio_doctor.wav`) is not distributed. | `scripts/synthesize_audio.py` |
| MOSS-TTS | Add the `clis/` folder of the MOSS-TTS repository to `PYTHONPATH`. | `scripts/synthesize_audio.py` |
| FFmpeg | On `PATH`, or set `FFMPEG` to its location. | `scripts/package_release.py`, evaluation scripts |
