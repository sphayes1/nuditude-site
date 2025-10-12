// Netlify Function: /.netlify/functions/generate
// POST: { prompt: string } -> { image: url }
// Tries multiple public models on Replicate for best compatibility.

const API_PREDICTIONS = 'https://api.replicate.com/v1/predictions';
const API_MODEL_VERSIONS = (model) => `https://api.replicate.com/v1/models/${model}/versions`;
const MODELS = [
  'stability-ai/sdxl',
  'black-forest-labs/flux-schnell'
];

exports.handler = async function (event) {\n  const log = (...a) => { try { console.log(...a); } catch {} };
  const json = (status, obj) => ({ statusCode: status, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(obj) });

  if (event.httpMethod === 'OPTIONS') return { statusCode: 204 };
  if (event.httpMethod === 'GET') return json(200, { ok: true, usage: 'POST with { prompt: string }' });
  if (event.httpMethod !== 'POST') return json(405, { error: 'Method Not Allowed' });

  try {
    const { prompt } = JSON.parse(event.body || '{}');
    if (!prompt || typeof prompt !== 'string') return json(400, { error: 'Missing prompt' });
    const token = process.env.REPLICATE_API_TOKEN;
    if (!token) return json(500, { error: 'Missing REPLICATE_API_TOKEN' });

    const headers = { 'Authorization': `Token ${token}`, 'Content-Type': 'application/json' };
    const errors = [];

    for (const model of MODELS) {\n      log('try model', model);
      try {
        // 1) Fetch latest version for this model
        const vres = await fetch(API_MODEL_VERSIONS(model), { headers: { 'Authorization': `Token ${token}` } });
        if (!vres.ok) {
          const t = await vres.text();
          errors.push({ model, step: 'versions', status: vres.status, detail: t });
          continue;
        }
        const vdata = await vres.json();
        const list = vdata.results || vdata.versions || [];
        const vid = (list[0] && (list[0].id || list[0].version || list[0].slug)) || (list.length ? list[list.length-1].id : null);
        if (!vid) { errors.push({ model, step: 'versions', status: 'no-version' }); continue; }

        // 2) Create prediction with minimal inputs for max compatibility
        log('creating prediction for', model, 'version', vid);\n        const createRes = await fetch(API_PREDICTIONS, {
          method: 'POST',
          headers,
          body: JSON.stringify({ version: vid, input: { prompt } })
        });
        if (!createRes.ok) {
          const text = await createRes.text();
          errors.push({ model, step: 'create', status: createRes.status, detail: text });
          continue;
        }

        const prediction = await createRes.json();\n        log('prediction created', prediction && prediction.id ? prediction.id : 'no-id');
        const url = prediction.urls?.get;
        if (!url) { errors.push({ model, step: 'create', status: 'no-url', detail: prediction }); continue; }

        // 3) Poll up to 2 minutes
        const started = Date.now();
        const timeoutMs = 120000;
        while (Date.now() - started < timeoutMs) {
          await new Promise(r => setTimeout(r, 2000));
          const poll = await fetch(url, { headers: { 'Authorization': `Token ${token}` } });
          const data = await poll.json();
          if (!poll.ok) { errors.push({ model, step: 'poll', status: poll.status, detail: data }); break; }
          if (data.status === 'succeeded') {
            const outputUrl = Array.isArray(data.output) ? data.output[0] : data.output;
            if (outputUrl) return json(200, { image: outputUrl, model });
            errors.push({ model, step: 'poll', status: 'no-output', detail: data });
            break;
          }
          if (data.status === 'failed' || data.status === 'canceled') {
            errors.push({ model, step: 'poll', status: data.status, detail: data });
            break;
          }
        }
      } catch (e) {
        errors.push({ model, step: 'exception', detail: String(e) });
      }
    }

    log('all attempts failed', errors);\n    return json(502, { error: 'All models failed', attempts: errors });
  } catch (err) {
    return json(500, { error: 'Server error', detail: String(err) });
  }
};