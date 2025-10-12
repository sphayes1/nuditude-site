// Netlify Function: /.netlify/functions/generate
// POST: { prompt: string } -> { image: url }
// Tries multiple public models on Replicate for best compatibility.
"use strict";

const API_PREDICTIONS = 'https://api.replicate.com/v1/predictions';
const API_MODEL_VERSIONS = function(model){ return 'https://api.replicate.com/v1/models/' + model + '/versions'; };
const MODELS = [ 'stability-ai/sdxl', 'black-forest-labs/flux-schnell' ];

async function callWorkerA1111(prompt) {
  const url = (process.env.WORKER_URL || '').replace(/\/$/, '') + '/sdapi/v1/txt2img';
  const body = {
    prompt: prompt,
    negative_prompt: 'blurry, deformed, extra limbs, watermark, text, logo, bad hands, bad anatomy',
    steps: 28,
    cfg_scale: 7,
    width: 768,
    height: 1024,
    sampler_name: 'DPM++ 2M Karras'
  };
  const res = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  if (!res.ok) {
    const t = await res.text();
    throw new Error('Worker error: ' + t);
  }
  const data = await res.json();
  const base64 = (data && data.images && data.images[0]) ? data.images[0] : null;
  if (!base64) throw new Error('Worker returned no image');
  return 'data:image/png;base64,' + base64;
}
exports.handler = async function (event) {
  const json = (status, obj) => ({ statusCode: status, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(obj) });

  if (event.httpMethod === 'OPTIONS') return { statusCode: 204 };
  if (event.httpMethod === 'GET') return json(200, { ok: true, usage: 'POST with { prompt: string }' });
  if (event.httpMethod !== 'POST') return json(405, { error: 'Method Not Allowed' });

  try {
    const parsed = JSON.parse(event.body || '{}');
    const prompt = (parsed && typeof parsed.prompt === 'string') ? parsed.prompt : '';
    if (!prompt) return json(400, { error: 'Missing prompt' });
    const token = process.env.REPLICATE_API_TOKEN;
    if (!token && !process.env.WORKER_URL) return json(500, { error: 'Missing REPLICATE_API_TOKEN or WORKER_URL' });
    if (process.env.WORKER_URL) { try { const image = await callWorkerA1111(prompt); return json(200, { image, model: 'worker-a1111' }); } catch(e){ /* fall through to Replicate */ } }

    const errors = [];
    for (let i = 0; i < MODELS.length; i++) {
      const model = MODELS[i];
      try {
        // 1) Fetch latest version
        const vres = await fetch(API_MODEL_VERSIONS(model), { headers: { 'Authorization': 'Token ' + token } });
        if (!vres.ok) {
          const t = await vres.text();
          errors.push({ model, step: 'versions', status: vres.status, detail: t });
          continue;
        }
        const vdata = await vres.json();
        const list = (vdata && (vdata.results || vdata.versions)) || [];
        const vid = list && list.length ? (list[0].id || list[0].version || list[0].slug) : null;
        if (!vid) { errors.push({ model, step: 'versions', status: 'no-version' }); continue; }

        // 2) Create prediction with minimal inputs
        const createRes = await fetch(API_PREDICTIONS, {
          method: 'POST',
          headers: {
            'Authorization': 'Token ' + token,
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({ version: vid, input: { prompt } })
        });
        if (!createRes.ok) {
          const text = await createRes.text();
          errors.push({ model, step: 'create', status: createRes.status, detail: text });
          continue;
        }

        const prediction = await createRes.json();
        const url = prediction && prediction.urls && prediction.urls.get;
        if (!url) { errors.push({ model, step: 'create', status: 'no-url', detail: prediction }); continue; }

        // 3) Poll
        const started = Date.now();
        const timeoutMs = 120000;
        while (Date.now() - started < timeoutMs) {
          await new Promise(function(r){ setTimeout(r, 2000); });
          const poll = await fetch(url, { headers: { 'Authorization': 'Token ' + token } });
          const data = await poll.json();
          if (!poll.ok) { errors.push({ model, step: 'poll', status: poll.status, detail: data }); break; }
          if (data && data.status === 'succeeded') {
            const out = Array.isArray(data.output) ? data.output[0] : data.output;
            if (out) return json(200, { image: out, model });
            errors.push({ model, step: 'poll', status: 'no-output', detail: data });
            break;
          }
          if (data && (data.status === 'failed' || data.status === 'canceled')) {
            errors.push({ model, step: 'poll', status: data.status, detail: data });
            break;
          }
        }
      } catch (e) {
        errors.push({ model, step: 'exception', detail: String(e) });
      }
    }

    return json(502, { error: 'All models failed', attempts: errors });
  } catch (err) {
    return json(500, { error: 'Server error', detail: String(err) });
  }
};