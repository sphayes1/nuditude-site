# Deploy on Cloudflare Pages (cheaper/free)

- Connect this GitHub repo to Cloudflare Pages → Create Project.
- Build command: none
- Output directory: NudieWebsite
- Functions: this repo includes `functions/api/generate.js` (maps to `/api/generate`).

After first deploy, add environment variables:
- RUNPOD_API_KEY = your RunPod API key
- RUNPOD_ENDPOINT_ID = your Serverless endpoint id

Then visit `/studio.html` and generate. The page calls `/api/generate` (Pages Function) and falls back to Netlify if not present.