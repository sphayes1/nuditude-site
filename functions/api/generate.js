// Cloudflare Pages Function: POST /api/generate
// Body: { prompt: string }
// Requires env: REPLICATE_API_TOKEN

const REPLICATE_API = "https://api.replicate.com/v1/models/stability-ai/sdxl/predictions";

export async function onRequestPost(context) {
  const { request, env } = context;

  try {
    const body = await request.json();
    const { prompt } = body;

    if (!prompt || typeof prompt !== "string") {
      return new Response(JSON.stringify({ error: "Missing prompt" }), {
        status: 400,
        headers: { "Content-Type": "application/json" }
      });
    }

    if (!env.REPLICATE_API_TOKEN) {
      return new Response(JSON.stringify({ error: "Missing REPLICATE_API_TOKEN" }), {
        status: 500,
        headers: { "Content-Type": "application/json" }
      });
    }

    const createRes = await fetch(REPLICATE_API, {
      method: "POST",
      headers: {
        "Authorization": `Token ${env.REPLICATE_API_TOKEN}`,
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
      return new Response(JSON.stringify({ error: "Replicate request failed", detail: text }), {
        status: createRes.status,
        headers: { "Content-Type": "application/json" }
      });
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
        headers: { "Authorization": `Token ${env.REPLICATE_API_TOKEN}` }
      });
      const data = await poll.json();

      if (!poll.ok) {
        return new Response(JSON.stringify({ error: "Polling failed", detail: data }), {
          status: poll.status,
          headers: { "Content-Type": "application/json" }
        });
      }

      if (data.status === "succeeded") {
        outputUrl = Array.isArray(data.output) ? data.output[0] : data.output;
        break;
      }

      if (data.status === "failed" || data.status === "canceled") {
        return new Response(JSON.stringify({ error: "Generation failed", detail: data }), {
          status: 500,
          headers: { "Content-Type": "application/json" }
        });
      }
    }

    if (!outputUrl) {
      return new Response(JSON.stringify({ error: "Generation timed out" }), {
        status: 504,
        headers: { "Content-Type": "application/json" }
      });
    }

    // Return CDN URL hosted by provider (no storage on our side)
    return new Response(JSON.stringify({ image: outputUrl }), {
      status: 200,
      headers: { "Content-Type": "application/json" }
    });
  } catch (err) {
    return new Response(JSON.stringify({ error: "Server error", detail: String(err) }), {
      status: 500,
      headers: { "Content-Type": "application/json" }
    });
  }
}
