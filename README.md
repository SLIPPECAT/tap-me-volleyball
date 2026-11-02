# Tap Me Volleyball

Tap yourself once in a phone-recorded amateur volleyball match, and get vertical (9:16) highlights of your rallies — the frame follows you, everyone else's face is blurred, and optional live-style commentary subtitles come in Korean, Japanese and English.

Built for the [Meta AI Developer Hackathon](https://cerebralvalley.ai/e/meta-ai-developer-hackathon) (Oct 26 – Nov 4, 2026) as a feature for Baegugu, a Korean amateur volleyball community. It runs as a mobile web app for Baegugu members; judges get access through the link in the submission form.

## What the user does

1. Open the app from the Baegugu banner and pick a match video. "Do it all for me" works 10 minutes at a time; for a longer match the user picks where to start, then taps "do the next 10 min too" on the results screen (the video is not uploaded again). "Pick one by one" scans up to 30 minutes.
2. Pause on a frame where they look big (on the near side of the court) and tap themselves. A quick check stops a tap on someone too small or on nobody. Optionally add a hint ("green jersey #10") and the team name.
3. Close the screen. If other people are ahead in line, the app shows how many and roughly when it starts. A push notification arrives when the highlights are ready: a list of small previews, a combined reel, and one-tap "save all to Photos".

## How Meta AI is used

Everything goes through the **Meta Model API**. Only face-blurred clips and stills are sent, uploaded files are deleted right after each answer, and the contributor (training) mode is never used.

| Step | Model | What it does |
|---|---|---|
| Find rallies | **Muse Spark** (`muse-spark-1.3`) | Watches the blurred match in 60-second chunks (2 fps) and returns when each rally starts and when the point is decided |
| Player cards | **SAM 3.1** (`sam-3.1`) | Detects every player at three moments of each rally |
| Find "me" in every rally | **Muse Spark** | Compares labeled player cards with a reference crop of the tapped user, one question per moment, and keeps the surest answer |
| Follow "me" | **SAM 3.1** | Video tracking of the chosen player across the whole rally, from just before the serve to just after the point (10 fps, up to 60 s); the vertical crop follows this track |
| Other players | **SAM 3.1** | Tracks everyone else, so face pixelation is never drawn on bodies and one team can be left unpixelated (with consent) |
| Team sorting (optional) | **Muse Spark** | Sorts players into the user's team and the opponents from two separate moments; a team is unpixelated only when both answers agree |
| Who won the point | **Muse Spark** | Asks which side serves the next rally (the rally winner serves next) and combines it with the side the user is on in that set |
| Commentary subtitles (optional) | **Muse Spark** | Writes 2–4 timed lines of broadcast-style commentary in ko/ja/en that match which team won the point |

Measured on the production server (GCP N2, 32 vCPU / 48 GB) with a 10-minute 1080p match (18 rallies, final run on Nov 2): rally finding 4 min, first highlight 9 min, 12 highlights plus the combined reel in 22 min, about $2.00 of Meta API usage. Most of that time is face detection on every output frame, which we keep on every frame so no face slips through. In earlier end-to-end runs the right player was picked in 23 of 24 rallies, and the scoring team was right in 14 of 16 rallies.

A highlight is made automatically only when Muse Spark is at least 0.9 sure **and** the user is on the near side of the court. Wrong picks found in a user check (0.75–0.85) were teammates with a similar build, and in the final run every uncertain pick (0.45–0.85) was on the far side, where players look small. Below 0.9 the app asks "is this you?"; far-side rallies are skipped and can still be picked by hand.

## Privacy by design

- Faces are blurred **before** anything is sent to Meta (OpenCV YuNet). Users can also blur on their own phone so the original never leaves the device.
- In the final video everyone except the tapped user is blurred (frontal and profile face detectors, held for 0.2 s so it does not flicker off). Users can turn it off or keep their own team unblurred, with consent.
- Clips uploaded to the Meta API are deleted right after each answer. On our server, all videos are deleted every day at 4:00 KST.
- Members only (Baegugu login), a consent screen before upload, and no videos of minors.

## Architecture

- `tapme/server.py` — FastAPI app: resumable uploads, jobs, push notifications, members-only gate, per-code limits, a fair render queue (jobs take turns), and a request log for operations.
- `tapme/scan.py`, `rally.py`, `rally_filter.py` — rally finding with Muse Spark.
- `tapme/auto.py` — "do it all for me": finds the user in every rally and queues a highlight for each.
- `tapme/track.py` — SAM 3.1 detection and video tracking.
- `tapme/jobs.py` — scene preparation, team sorting, scoring team, render orchestration.
- `tapme/render.py`, `anonymize.py` — 9:16 crop that follows the user, a marker at their feet, face pixelation, subtitles. Face detection runs in one shared pool of worker processes.
- `tapme/static/` — the mobile web app (installable PWA) and on-device blurring (`browser-blur.js`, ONNX Runtime Web).
- `deploy/` — Caddy config, an idle-stop script, and a small Cloud Run service that wakes the VM when someone opens the app.

## Run locally

```bash
uv sync
echo "MODEL_API_KEY=..." > .env   # Meta Model API key
uv run uvicorn tapme.server:app --port 8000
uv run pytest                     # tests (no API calls)
uv run ruff check .
```

The YuNet model file goes in `tapme/models/` (see `tapme/anonymize.py`).

## Third-party components

| Component | Use | License |
|---|---|---|
| [YuNet face detector](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) (OpenCV Zoo) | Face blurring on the server and in the browser | MIT |
| OpenCV (`opencv-python-headless`) | Video I/O, face detection | Apache-2.0 |
| OpenCV profile-face Haar cascade | Profile faces in the final video | Intel License Agreement for OpenCV |
| [`meta-sam-parser`](https://pypi.org/project/meta-sam-parser/) | Parsing SAM 3.1 API responses | SAM License |
| ONNX Runtime Web · mp4box.js · mp4-muxer (via jsDelivr) | On-device blurring in the browser | MIT · BSD-3-Clause · MIT |
| FastAPI · Uvicorn · Pillow · NumPy · python-dotenv | Server | MIT · BSD-3-Clause · MIT-CMU · BSD-3-Clause · BSD-3-Clause |
| OpenAI Python SDK (Meta Model API client) · python-multipart | Server | Apache-2.0 |
| pywebpush | Push notifications | MPL-2.0 |

## How it was built

All product code in this repository was written during the hackathon (Oct 26 – Nov 4, 2026), with Claude Code as a coding assistant. Experiments, measurements and decisions are recorded in a separate experiment repository.

## License

© 2026 Baegugu. All rights reserved. This repository is shared publicly for hackathon judging; no open-source license is granted. Third-party components keep their own licenses (see above).
