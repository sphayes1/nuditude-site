// Netlify Function: POST /.netlify/functions/generate
// Body: { prompt: string }
// Requires env: REPLICATE_API_TOKEN

// Use the model-specific endpoint so we don't need a hardcoded version ID
const REPLICATE_API = "https://api.replicate.com/v1/models/stability-ai/sdxl/predictions";

exports.handler = async function (event) {
  if (event.httpMethod !== "POST") {
    return { statusCode: 405, body: "Method Not Allowed" };
  }

  try {
    const { prompt } = JSON.parse(event.body || "{}");
    if (!prompt || typeof prompt !== "string") {
      return { statusCode: 400, body: JSON.stringify({ error: "Missing prompt" }) };
    }

    if (!process.env.REPLICATE_API_TOKEN) {
      return { statusCode: 500, body: JSON.stringify({ error: "Missing REPLICATE_API_TOKEN" }) };
    }

    const createRes = await fetch(REPLICATE_API, {
      method: "POST",
      headers: {
        "Authorization": `Token ${process.env.REPLICATE_API_TOKEN}`,
        "Content-Type": "application/json"
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
      return { statusCode: createRes.status, body: JSON.stringify({ error: "Replicate request failed", detail: text }) };
    }

    const prediction = await createRes.json();
    const url = prediction.urls?.get;
    
    // Poll for completion
    const started = Date.now();
    const timeoutMs = 90000; // 90s
    let outputUrl = null;
    while (Date.now() - started < timeoutMs) {
      await new Promise(r => setTimeout(r, 2000));
      const poll = await fetch(url, {
        headers: { "Authorization": `Token ${process.env.REPLICATE_API_TOKEN}` }
      });
      const data = await poll.json();
      if (!poll.ok) {
        return { statusCode: poll.status, body: JSON.stringify({ error: "Polling failed", detail: data }) };
      }
      if (data.status === "succeeded") {
        outputUrl = Array.isArray(data.output) ? data.output[0] : data.output;
        break;
      }
      if (data.status === "failed" || data.status === "canceled") {
        return { statusCode: 500, body: JSON.stringify({ error: "Generation failed", detail: data }) };
      }
    }

    if (!outputUrl) {
      return { statusCode: 504, body: JSON.stringify({ error: "Generation timed out" }) };
    }

    // Return CDN URL hosted by provider (no storage on our side)
    return {
      statusCode: 200,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: outputUrl })
    };
  } catch (err) {
    return { statusCode: 500, body: JSON.stringify({ error: "Server error", detail: String(err) }) };
  }
};
