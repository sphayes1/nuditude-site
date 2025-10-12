export async function onRequestPost(ctx) {
  const { request, env } = ctx;
  const json = (s, o) => new Response(JSON.stringify(o), { status: s, headers: { 'Content-Type': 'application/json' } });

  try {
    const body = await request.json().catch(() => ({}));
    const prompt = typeof body.prompt === 'string' ? body.prompt : '';
    if (!prompt) return json(400, { error: 'Missing prompt' });

    const endpoint = env.RUNPOD_ENDPOINT_ID;
    const key = env.RUNPOD_API_KEY;
    if (!endpoint || !key) return json(500, { error: 'Missing RUNPOD env' });

    const base = `https://api.runpod.ai/v2/${endpoint}`;
    const headers = { 'Authorization': `Bearer ${key}`, 'Content-Type': 'application/json' };
    const createBody = {
      input: {
        prompt,
        negative_prompt: 'blurry, deformed, extra limbs, watermark, text, logo, bad hands, bad anatomy',
        width: 768,
        height: 1024,
        steps: 28,
        cfg_scale: 7,
      }
    };

    const create = await fetch(`${base}/run`, { method: 'POST', headers, body: JSON.stringify(createBody) });
    if (!create.ok) {
      const t = await create.text();
      return json(create.status, { error: 'RunPod create failed', detail: t });
    }
    const created = await create.json();
    const id = created && (created.id || created.jobId || created['id']);
    if (!id) return json(500, { error: 'RunPod: missing job id', detail: created });

    const started = Date.now();
    const timeoutMs = 180000;
    while (Date.now() - started < timeoutMs) {
      await new Promise(r => setTimeout(r, 2000));
      const statusRes = await fetch(`${base}/status/${id}`, { headers });
      if (!statusRes.ok) {
        const t = await statusRes.text();
        return json(statusRes.status, { error: 'RunPod status failed', detail: t });
      }
      const status = await statusRes.json();
      const s = status && (status.status || status.state || (status.execution && status.execution.status));
      if (s === 'COMPLETED') {
        const out = status.output || (status.execution && status.execution.output) || {};
        const img = out.image || (out.images && out.images[0]);
        if (!img) return json(500, { error: 'RunPod: no image in output', detail: out });
        const image = img.startsWith('data:image') ? img : `data:image/png;base64,${img}`;
        return json(200, { image, provider: 'runpod-pages' });
      }
      if (s === 'FAILED') {
        const detail=(status && ((status.output && status.output.detail) || status.detail)) || status; return json(500, { error: 'RunPod job failed', detail });
      }
    }
    return json(504, { error: 'RunPod: job timed out' });
  } catch (e) {
    return json(500, { error: 'Server error', detail: String(e) });
  }
}
