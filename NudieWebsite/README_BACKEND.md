# Image Creation & Serving – Options

This repo includes a minimal serverless path to generate and serve images without running your own GPU. You can upgrade later to a dedicated GPU worker.

## Option A: Serverless + Hosted Model (Recommended to start)

- File: `serverless/netlify/functions/generate.js`
- Provider: Replicate SDXL (hosted inference)
- Deploy: Netlify (Functions enabled)
- Frontend demo: `studio.html`

Steps:
1. Create a Netlify site pointing to this folder.
2. In Site settings → Environment variables, add `REPLICATE_API_TOKEN`.
3. Deploy. The studio page calls `/.netlify/functions/generate` to get an image URL (served by the provider CDN). Nothing is stored on your site.

Notes:
- This is private by design: no uploads stored here; only prompt text is sent to the function.
- Add a paywall or rate limit before exposing widely.

## Option B: Full Pipeline (Selfie → Analyze → Enhance → Generate)

When you’re ready for selfie‑to‑style, run a GPU worker (RunPod/Modal). Recommended architecture:

- `frontend` (public): upload → gets presigned PUT URL to object storage (S3/R2)
- `backend` (API): creates presigned URLs, enqueues jobs, returns job id
- `worker` (GPU): pulls job, downloads input from presigned URL, runs pipeline (analyze → enhance → generate), uploads result to storage, marks job done
- Client polls `/status/:id` to retrieve a presigned GET URL for the result

Safety:
- Age‑gate, explicit consent text, block celebrities/minors.
- Auto‑delete inputs/outputs on TTL (e.g., 24h lifecycle policy on bucket).

## Storage & Serving

- Start simple: serve provider CDN URLs (Replicate returns them). No data retention.
- Upgrade: store outputs in S3/R2 with presigned GET, set TTL lifecycle policy.

## Stripe/Gumroad Paywall (Optional)

- Gate the studio behind Stripe Payment Links or Gumroad, then reveal `studio.html` via a unique link or email after purchase.


## Option C: Self-hosted worker (allows adult content)

Use an Automatic1111 (Stable Diffusion WebUI) pod on RunPod. You control safety filters and model choice. The site calls your worker if `WORKER_URL` is set.

Steps:
1) Create RunPod account and launch an A1111 template (20GB+ GPU). Enable public HTTP port 7860.
2) In WebUI, enable API (Settings → API). Load an SDXL model you are licensed to use.
3) Copy your public endpoint base URL (e.g., https://your-pod-1234.runpod.run).
4) In Netlify → Site settings → Environment variables, add:
   - WORKER_URL = https://your-pod-1234.runpod.run
5) Redeploy. The function will prefer WORKER_URL and return a `data:image/png;base64,...` URL.

Legal & Safety:
- Adults only. No minors, celebrities, or non‑consensual likeness.
- Display an age gate and consent text. Keep logs minimal.
- If you store images, enforce auto‑deletion and user controls.
