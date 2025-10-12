// Netlify Function: /.netlify/functions/generate
// POST: { prompt: string } -> { image: url }
// Uses Replicate: dynamically fetches latest SDXL version, then creates a prediction.

const API_PREDICTIONS = 'https://api.replicate.com/v1/predictions';
const API_VERSIONS = 'https://api.replicate.com/v1/models/stability-ai/sdxl/versions';

exports.handler = async function (event) {
  const json = (status, obj) => ({ statusCode: status, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(obj) });

  if (event.httpMethod === 'OPTIONS') return { statusCode: 204 };
  if (event.httpMethod === 'GET') return json(200, { ok: true, usage: 'POST with { prompt: string }' });
  if (event.httpMethod !== 'POST') return json(405, { error: 'Method Not Allowed' });

  try {
    const { prompt } = JSON.parse(event.body || '{}');
    if (!prompt || typeof prompt !== 'string') return json(400, { error: 'Missing prompt' });
    if (!process.env.REPLICATE_API_TOKEN) return json(500, { error: 'Missing REPLICATE_API_TOKEN' });

    // 1) Get latest SDXL version id
    const vres = await fetch(API_VERSIONS, { headers: { 'Authorization': `Token ${process.env.REPLICATE_API_TOKEN}` } });
    if (!vres.ok) {
      const t = await vres.text();
      return json(vres.status, { error: 'Failed to fetch SDXL versions', detail: t });
    }
    const vdata = await vres.json();
    const list = vdata.results || vdata.versions || [];
    const vid = (list[0] && (list[0].id || list[0].version || list[0].slug)) || (list.length ? list[list.length-1].id : null);
    if (!vid) return json(500, { error: 'No SDXL version available' });

    // 2) Create prediction
    const createRes = await fetch(API_PREDICTIONS, {
      method: 'POST',
      headers: {
        'Authorization': `Token ${process.env.REPLICATE_API_TOKEN}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        version: vid,
        input: {
          prompt,
          width: 768,
          height: 1024,
          num_inference_steps: 30,
          guidance_scale: 7.5
        }
      })
    });
    if (!createRes.ok) {
      const text = await createRes.text();
      return json(createRes.status, { error: 'Replicate request failed', detail: text });
    }
    const prediction = await createRes.json();
    const url = prediction.urls?.get;

    // 3) Poll
    const started = Date.now();
    const timeoutMs = 120000; // 2m
    let outputUrl = null;
    while (Date.now() - started < timeoutMs) {
      await new Promise(r => setTimeout(r, 2000));
      const poll = await fetch(url, { headers: { 'Authorization': `Token ${process.env.REPLICATE_API_TOKEN}` } });
      const data = await poll.json();
      if (!poll.ok) return json(poll.status, { error: 'Polling failed', detail: data });
      if (data.status === 'succeeded') {
        outputUrl = Array.isArray(data.output) ? data.output[0] : data.output;
        break;
      }
      if (data.status === 'failed' || data.status === 'canceled') {
        return json(500, { error: 'Generation failed', detail: data });
      }
    }

    if (!outputUrl) return json(504, { error: 'Generation timed out' });
    return json(200, { image: outputUrl });
  } catch (err) {
    return json(500, { error: 'Server error', detail: String(err) });
  }
};