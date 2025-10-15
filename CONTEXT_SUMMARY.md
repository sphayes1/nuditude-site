# Project Context Snapshot _(generated October 13, 2025)_

This document fast-tracks a fresh context window to the current state of the **Nuditude** codebase and recent collaboration history.

---

## Repositories & Deploy Targets
- **Repo**: `sphayes1/nuditude-site` (branch `main`, latest commit `fce91f2 Add segmentation scaffold and roadmap`).
- **RunPod Worker Container**: `spencer45/runpod-sdxl-ipadapter:faceid-latest`
  - **Latest push digest**: `sha256:d14f20a167331faab07fb3566b1cdcfde514632b5a5a8dc9b902188f9cea5c9c`
  - Built with SDXL inpainting + FaceID + **YOLOv8 + SAM segmentation** + **mandatory detection enforcement** + **JSON serialization fixes**.
  - **Enable segmentation**: Set `USE_SEGMENTATION=1` in RunPod template environment variables.
- **Cloudflare Pages project**: `nuditude-site` (functions in repo root `functions/api/*`).
- **Model checkpoints** cached inside the worker:
  - InsightFace `buffalo_l` (face detection, landmarks, recognition)
  - SAM ViT-B checkpoint (~375MB, downloaded at build time)
  - YOLOv8 nano (auto-downloaded on first run)

---

## Key Files Recently Touched
- `runpod-ipadapter/segmentation.py` — **NEW**: YOLOv8 + SAM segmentation module with DeepLabV3 fallback.
- `runpod-ipadapter/handler.py` — RunPod serverless handler with 3-tier mask fallback (YOLOv8+SAM → DeepLabV3 → heuristic).
- `runpod-ipadapter/Dockerfile` — Updated to download SAM checkpoint at build time.
- `runpod-ipadapter/requirements.txt` — Added ultralytics, segment-anything, opencv-python-headless.
- `IMPLEMENTATION_ROADMAP.md` — **NEW**: Phase-by-phase breakdown of roadmap milestones.
- `functions/api/generate.js` — Cloudflare Pages function invoking RunPod; enforces reference image, logs masks.
- `functions/api/prompt.js` — Admin API storing master/negative prompts, default steps/guidance, generation logs in KV.
- `NudieWebsite/admin.html` — Password-gated prompt editor + log viewer (shows input/mask/output thumbnails).
- `NudieWebsite/assets/studio.js` — Frontend studio flow uploading reference photo, calling analyze/generate endpoints.

---

## Major Changes Delivered
1. **🚀 YOLOv8 + SAM Segmentation Implementation** _(October 13-14, 2025)_
   - Implemented full YOLOv8 person detection + SAM mask refinement pipeline.
   - Three-tier fallback strategy: YOLOv8+SAM → DeepLabV3 → heuristic rectangle.
   - **⚠️ CRITICAL POLICY: Jobs are auto-rejected if no face OR person detected** (enforced in handler).
   - SAM ViT-B checkpoint (~375MB) bundled in Docker image.
   - Comprehensive diagnostics logged (confidence scores, bounding boxes, mask coverage).
   - Enable with `USE_SEGMENTATION=1` environment variable in RunPod.
   - Tunable via environment variables: `YOLO_CONFIDENCE_THRESHOLD`, `SAM_MODEL_TYPE`, etc.
   - **Fixed YOLOv8+SAM availability detection** using direct module attribute access (October 14).
   - **Fixed JSON serialization errors** for numpy types in response payload (October 14).

2. **RunPod Worker Upgrade**
   - Swapped to `StableDiffusionXLInpaintPipeline` + IP-Adapter FaceID.
   - Auto-generates clothing masks (face-preserving, feathered).
   - Logs mask image and prompt metadata in responses.
   - Removed duplicate `generator` argument to prevent diffusers TypeError.

3. **Prompt Management & Logging**
   - Added `/admin.html` with password gate (`ADMIN_PASSWORD`) and KV-backed storage (`PROMPT_STORE`).
   - Exposes master prompt, negative prompt, default steps/guidance, allow-user flag.
   - Log table shows reference, mask, output thumbnails, prompts, settings.

4. **API Hardening**
   - `generate.js` now requires a reference image (supports both `reference_image` and `referenceImage` keys).
   - Job payloads log to KV with auto mask preview.
   - User prompt acceptance toggled via admin flag + stored default values.

5. **Master Prompt Controls**
   - Master prompt auto-prepends to user prompt; default negative prompt & steps/guidance stored in KV.
   - Admin page can refresh logs, display saved values immediately after saving.

6. **Docker Build Automation**
   - Rebuilt/pushed RunPod image multiple times during refactors (latest digest above).

---

## Common Commands / Paths
- Build RunPod image:  
  `docker build -t runpod-ipadapter:latest ./runpod-ipadapter`
- Push to Docker Hub:  
  `docker push spencer45/runpod-sdxl-ipadapter:faceid-latest`
- Cloudflare Pages deploy: connect GitHub repo `nuditude-site`, ensure functions directory `functions/`.
- Admin prompt editor: `https://nuditude-site.pages.dev/admin.html`

---

## Feature Requests & Tasks Fulfilled for the User
- **✅ Implement full YOLOv8 + SAM segmentation** _(completed October 13, 2025)_
- **✅ Enforce mandatory face/person detection policy** _(completed October 13, 2025)_
- Fix Python syntax errors in `handler.py` logging.
- Rebuild & push RunPod worker images on demand.
- Implement master prompt system with admin UI.
- Add allow-user-prompt flag + environment toggle.
- Surface generation logs (input/output/settings).
- Transition to inpainting workflow with auto mask.
- Require reference image & accept camelCase payloads.
- Provide tailored prompt/negative prompt examples (e.g., swimsuit transformation).
- Explain FaceID identity preservation; troubleshoot poor outputs.

---

## Outstanding Considerations
- Admin UI relies on `PROMPT_STORE` KV binding and `ADMIN_PASSWORD` env var in Cloudflare Pages.
- ~~Inpainting mask is heuristic (torso rectangle below face)~~ → **RESOLVED**: YOLOv8 + SAM now available.
- Master prompt length currently triggers CLIP truncation warnings—consider shortening high-detail phrases.
- **⚠️ CRITICAL POLICY**: All jobs require face OR person detection — enforced in handler, documented in roadmap.
- **✅ DEPLOYED**: Docker image rebuilt with YOLOv8+SAM availability fix and JSON serialization fix (digest `d14f20a...`).
- **Next step**: Update RunPod template with new image digest (`sha256:d14f20a167331faab07fb3566b1cdcfde514632b5a5a8dc9b902188f9cea5c9c`) and verify YOLOv8+SAM is working correctly.

---

*End of context snapshot — safe to delete/regenerate after major architectural changes.* 
