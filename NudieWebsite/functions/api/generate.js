const json = (status, body) =>
  new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });

const PROMPT_KEY = 'master_prompt';
const NEGATIVE_KEY = 'negative_prompt';
const ALLOW_KEY = 'allow_user_prompt';

const parseBool = (value, fallback = false) => {
  if (typeof value === 'boolean') return value;
  if (typeof value === 'number') return value !== 0;
  if (typeof value === 'string') {
    const normalized = value.trim().toLowerCase();
    if (['true', '1', 'yes', 'on'].includes(normalized)) return true;
    if (['false', '0', 'no', 'off'].includes(normalized)) return false;
  }
  return fallback;
};

const parseNumber = (value, fallback) => {
  const num = Number(value);
  return Number.isFinite(num) ? num : fallback;
};

async function loadPromptConfig(env) {
  if (!env.PROMPT_STORE) {
    return {
      masterPrompt: '',
      negativePrompt: '',
      allowUserPrompt: false,
    };
  }

  const [masterPrompt, negativePrompt, allowValue] = await Promise.all([
    env.PROMPT_STORE.get(PROMPT_KEY),
    env.PROMPT_STORE.get(NEGATIVE_KEY),
    env.PROMPT_STORE.get(ALLOW_KEY),
  ]);

  return {
    masterPrompt: masterPrompt || '',
    negativePrompt: negativePrompt || '',
    allowUserPrompt: allowValue === 'true',
  };
}

export async function onRequestPost(context) {
  const { request, env } = context;

  try {
    const body = await request.json().catch(() => ({}));
    const userPrompt = typeof body.prompt === 'string' ? body.prompt.trim() : '';
    if (!userPrompt) {
      return json(400, { error: 'Missing prompt' });
    }

    const endpoint = env.RUNPOD_ENDPOINT_ID;
    const key = env.RUNPOD_API_KEY;
    if (!endpoint || !key) {
      return json(500, { error: 'Missing RUNPOD configuration' });
    }

    const promptConfig = await loadPromptConfig(env);
    const allowUserPrompt = parseBool(body.allowUserPrompt, promptConfig.allowUserPrompt);

    const negativePromptOverride =
      typeof body.negativePrompt === 'string' && body.negativePrompt.trim() !== ''
        ? body.negativePrompt.trim()
        : promptConfig.negativePrompt;

    const width = parseNumber(body.width, 768);
    const height = parseNumber(body.height, 1024);
    const steps = parseNumber(body.num_inference_steps ?? body.steps, 28);
    const guidance = parseNumber(body.guidance_scale ?? body.cfg_scale, 7);
    const ipAdapterScale = parseNumber(body.ip_adapter_scale, 0.8);

    const referenceImage = typeof body.reference_image === 'string' ? body.reference_image : undefined;
    const seed = body.seed !== undefined ? String(body.seed) : undefined;

    if (!userPrompt && !promptConfig.masterPrompt) {
      return json(400, { error: 'Prompt required', detail: 'Provide a user prompt or configure a master prompt.' });
    }

    const createBody = {
      input: {
        prompt: userPrompt,
        negative_prompt: negativePromptOverride || undefined,
        master_prompt: promptConfig.masterPrompt,
        use_master_prompt: Boolean(promptConfig.masterPrompt),
        allow_user_prompt: allowUserPrompt,
        width,
        height,
        num_inference_steps: steps,
        guidance_scale: guidance,
        ip_adapter_scale: ipAdapterScale,
      },
    };

    if (referenceImage) {
      createBody.input.reference_image = referenceImage;
    }
    if (seed !== undefined) {
      createBody.input.seed = seed;
    }

    const base = `https://api.runpod.ai/v2/${endpoint}`;
    const headers = {
      Authorization: `Bearer ${key}`,
      'Content-Type': 'application/json',
    };

    const create = await fetch(`${base}/run`, {
      method: 'POST',
      headers,
      body: JSON.stringify(createBody),
    });
    if (!create.ok) {
      const errorDetail = await create.text();
      return json(create.status, { error: 'RunPod create failed', detail: errorDetail });
    }

    const created = await create.json();
    const id = created && (created.id || created.jobId || created['id']);
    if (!id) {
      return json(500, { error: 'RunPod: missing job id', detail: created });
    }

    const started = Date.now();
    const timeoutMs = Number.isFinite(body.timeoutMs) ? Number(body.timeoutMs) : 180000;

    while (Date.now() - started < timeoutMs) {
      await new Promise((resolve) => setTimeout(resolve, 2000));
      const statusRes = await fetch(`${base}/status/${id}`, { headers });
      if (!statusRes.ok) {
        const detail = await statusRes.text();
        return json(statusRes.status, { error: 'RunPod status failed', detail });
      }

      const status = await statusRes.json();
      const state = status && (status.status || status.state || (status.execution && status.execution.status));
      if (state === 'COMPLETED') {
        const out = status.output || (status.execution && status.execution.output) || {};
        const img = out.image || (out.images && out.images[0]);
        if (!img) {
          return json(500, { error: 'RunPod: no image in output', detail: out });
        }
        const image = img.startsWith('data:image') ? img : `data:image/png;base64,${img}`;
        return json(200, {
          image,
          provider: 'runpod-pages',
        });
      }

      if (state === 'FAILED') {
        return json(500, { error: 'RunPod job failed', detail: status });
      }
    }

    return json(504, { error: 'RunPod: job timed out' });
  } catch (error) {
    return json(500, { error: 'Server error', detail: String(error) });
  }
}
