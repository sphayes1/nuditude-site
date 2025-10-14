# Project Context Snapshot _(generated October 13, 2025)_

This document fast-tracks a fresh context window to the current state of the **Nuditude** codebase and recent collaboration history.

---

## Repositories & Deploy Targets
- **Repo**: `sphayes1/nuditude-site` (branch `main`, latest commit `52d6a22 Avoid double generator in FaceID inpaint path`).
- **RunPod Worker Container**: `spencer45/runpod-sdxl-ipadapter:faceid-latest`  
  - Latest push digest: `sha256:2894a8198c2c49070ff082e9e6e21410a2cf0b2e10ddb13ecdf5d2fc639e4add`  
  - Built with SDXL inpainting + FaceID auto-mask workflow.
- **Cloudflare Pages project**: `nuditude-site` (functions in repo root `functions/api/*`).
- **InsightFace models** cached inside the worker: `buffalo_l` suite (detection, landmarks, recognition).

---

## Key Files Recently Touched
- `runpod-ipadapter/handler.py` — RunPod serverless handler (now SDXL inpainting, auto clothing mask, FaceID support).
- `functions/api/generate.js` — Cloudflare Pages function invoking RunPod; enforces reference image, logs masks.
- `functions/api/prompt.js` — Admin API storing master/negative prompts, default steps/guidance, generation logs in KV.
- `NudieWebsite/admin.html` — Password-gated prompt editor + log viewer (shows input/mask/output thumbnails).
- `NudieWebsite/assets/studio.js` — Frontend studio flow uploading reference photo, calling analyze/generate endpoints.

---

## Major Changes Delivered
1. **RunPod Worker Upgrade**
   - Swapped to `StableDiffusionXLInpaintPipeline` + IP-Adapter FaceID.
   - Auto-generates clothing masks (face-preserving, feathered).
   - Logs mask image and prompt metadata in responses.
   - Removed duplicate `generator` argument to prevent diffusers TypeError.

2. **Prompt Management & Logging**
   - Added `/admin.html` with password gate (`ADMIN_PASSWORD`) and KV-backed storage (`PROMPT_STORE`).
   - Exposes master prompt, negative prompt, default steps/guidance, allow-user flag.
   - Log table shows reference, mask, output thumbnails, prompts, settings.

3. **API Hardening**
   - `generate.js` now requires a reference image (supports both `reference_image` and `referenceImage` keys).
   - Job payloads log to KV with auto mask preview.
   - User prompt acceptance toggled via admin flag + stored default values.

4. **Master Prompt Controls**
   - Master prompt auto-prepends to user prompt; default negative prompt & steps/guidance stored in KV.
   - Admin page can refresh logs, display saved values immediately after saving.

5. **Docker Build Automation**
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
- Inpainting mask is heuristic (torso rectangle below face); may benefit from future segmentation upgrades.
- Master prompt length currently triggers CLIP truncation warnings—consider shortening high-detail phrases.

---

*End of context snapshot — safe to delete/regenerate after major architectural changes.* 
