// Netlify Function: /.netlify/functions/generate
// Methods:
//  - GET: returns usage JSON
//  - POST: { prompt: string } -> { image: url }
// Requires env: REPLICATE_API_TOKEN

const REPLICATE_API = 'https://api.replicate.com/v1/models/stability-ai/sdxl/predictions';

exports.handler = async function (event) {
  const json = (status, obj) => ({
    statusCode: status,
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(obj)
  });

  if (event.httpMethod === 'OPTIONS') {
    return { statusCode: 204 };
  }
  if (event.httpMethod === 'GET') {
    return json(200, { ok: true, usage: 'POST with { prompt: string }' });
  }
  if (event.httpMethod !== 'POST') {
    return json(405, { error: 'Method Not Allowed' });
  }

  try {
    const { prompt } = JSON.parse(event.body || '{}');
    if (!prompt || typeof prompt !== 'string') {
      return json(400, { error: 'Missing prompt' });
    }
    if (!process.env.REPLICATE_API_TOKEN) {
      return json(500, { error: 'Missing REPLICATE_API_TOKEN' });
    }

    const createRes = await fetch(REPLICATE_API, {
      method: 'POST',
      headers: {
        'Authorization': `Token ${process.env.REPLICATE_API_TOKEN}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
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

    const started = Date.now();
    const timeoutMs = 90000;
    let outputUrl = null;
    while (Date.now() - started < timeoutMs) {
      await new Promise(r => setTimeout(r, 2000));
      const poll = await fetch(url, {
        headers: { 'Authorization': `Token ${process.env.REPLICATE_API_TOKEN}` }
      });
      const data = await poll.json();
      if (!poll.ok) {
        return json(poll.status, { error: 'Polling failed', detail: data });
      }
      if (data.status === 'succeeded') {
        outputUrl = Array.isArray(data.output) ? data.output[0] : data.output;
        break;
      }
      if (data.status === 'failed' || data.status === 'canceled') {
        return json(500, { error: 'Generation failed', detail: data });
      }
    }

    if (!outputUrl) {
      return json(504, { error: 'Generation timed out' });
    }

    return json(200, { image: outputUrl });
  } catch (err) {
    return json(500, { error: 'Server error', detail: String(err) });
  }
};