const json = (status, body) =>
  new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });

const PROMPT_KEY = 'master_prompt';
const NEGATIVE_KEY = 'negative_prompt';
const ALLOW_KEY = 'allow_user_prompt';
const STEPS_KEY = 'default_steps';
const GUIDANCE_KEY = 'default_guidance';
const STRENGTH_KEY = 'default_strength';
const SEED_KEY = 'default_seed';
const FACE_PADDING_KEY = 'face_padding';
const CONTROLNET_ENABLED_KEY = 'controlnet_enabled';
const CONTROLNET_TYPE_KEY = 'controlnet_type';
const CONTROLNET_SCALE_KEY = 'controlnet_scale';
const FACEID_ENABLED_KEY = 'faceid_enabled';
const FACEID_SCALE_KEY = 'faceid_scale';
const LOGS_KEY = 'generation_logs';
const MAX_LOG_ENTRIES = 25;
// Optimized defaults for anatomy preservation and quality
const DEFAULT_STEPS = 50;  // Higher for complete inference, better quality
const DEFAULT_GUIDANCE = 10;  // Higher for sharper erotic details (9-12 recommended)
const DEFAULT_STRENGTH = 0.45;  // Lower to preserve anatomy, prevent over-inpainting
const DEFAULT_SEED = -1;  // -1 = random
const DEFAULT_FACE_PADDING = 0.05;  // 0.15-0.20 for full outfit swaps

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
      defaultSteps: DEFAULT_STEPS,
      defaultGuidance: DEFAULT_GUIDANCE,
      defaultStrength: DEFAULT_STRENGTH,
      defaultSeed: DEFAULT_SEED,
      facePadding: DEFAULT_FACE_PADDING,
    };
  }

  const [masterPrompt, negativePrompt, allowValue, stepsValue, guidanceValue, strengthValue, seedValue, facePaddingValue, controlnetEnabledValue, controlnetTypeValue, controlnetScaleValue, faceidEnabledValue, faceidScaleValue] = await Promise.all([
    env.PROMPT_STORE.get(PROMPT_KEY),
    env.PROMPT_STORE.get(NEGATIVE_KEY),
    env.PROMPT_STORE.get(ALLOW_KEY),
    env.PROMPT_STORE.get(STEPS_KEY),
    env.PROMPT_STORE.get(GUIDANCE_KEY),
    env.PROMPT_STORE.get(STRENGTH_KEY),
    env.PROMPT_STORE.get(SEED_KEY),
    env.PROMPT_STORE.get(FACE_PADDING_KEY),
    env.PROMPT_STORE.get(CONTROLNET_ENABLED_KEY),
    env.PROMPT_STORE.get(CONTROLNET_TYPE_KEY),
    env.PROMPT_STORE.get(CONTROLNET_SCALE_KEY),
    env.PROMPT_STORE.get(FACEID_ENABLED_KEY),
    env.PROMPT_STORE.get(FACEID_SCALE_KEY),
  ]);

  return {
    masterPrompt: masterPrompt || '',
    negativePrompt: negativePrompt || '',
    allowUserPrompt: allowValue === 'true',
    defaultSteps: stepsValue ? Number(stepsValue) : DEFAULT_STEPS,
    defaultGuidance: guidanceValue ? Number(guidanceValue) : DEFAULT_GUIDANCE,
    defaultStrength: strengthValue ? Number(strengthValue) : DEFAULT_STRENGTH,
    defaultSeed: seedValue ? Number(seedValue) : DEFAULT_SEED,
    facePadding: facePaddingValue ? Number(facePaddingValue) : DEFAULT_FACE_PADDING,
    controlnetEnabled: controlnetEnabledValue === null ? true : controlnetEnabledValue === 'true',
    controlnetType: controlnetTypeValue || 'openpose',
    controlnetScale: controlnetScaleValue ? Number(controlnetScaleValue) : 0.5,
    faceidEnabled: faceidEnabledValue === null ? false : faceidEnabledValue === 'true',
    faceidScale: faceidScaleValue ? Number(faceidScaleValue) : 0.55,
  };
}

async function appendLog(env, entry) {
  if (!env.PROMPT_STORE) {
    return;
  }
  try {
    const rawLogs = await env.PROMPT_STORE.get(LOGS_KEY);
    let logs = [];
    if (rawLogs) {
      logs = JSON.parse(rawLogs);
      if (!Array.isArray(logs)) {
        logs = [];
      }
    }
    logs.unshift(entry);
    if (logs.length > MAX_LOG_ENTRIES) {
      logs = logs.slice(0, MAX_LOG_ENTRIES);
    }
    await env.PROMPT_STORE.put(LOGS_KEY, JSON.stringify(logs));
  } catch (error) {
    console.error('Failed to append log entry:', error);
  }
}

const ensureDataUrl = (image, defaultMime = 'image/png') => {
  if (!image || typeof image !== 'string') {
    return undefined;
  }
  if (image.startsWith('data:image')) {
    return image;
  }
  if (image.startsWith('http://') || image.startsWith('https://')) {
    return image;
  }
  return `data:${defaultMime};base64,${image}`;
};

export async function onRequestPost(context) {
  const { request, env } = context;

  try {
    const body = await request.json().catch(() => ({}));
    const userPrompt = typeof body.prompt === 'string' ? body.prompt.trim() : '';

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
    const steps = parseNumber(body.num_inference_steps ?? body.steps, promptConfig.defaultSteps);
    const guidance = parseNumber(body.guidance_scale ?? body.cfg_scale, promptConfig.defaultGuidance);
    const strength = parseNumber(body.strength, promptConfig.defaultStrength);
    const ipAdapterScale = parseNumber(body.ip_adapter_scale, 0.55);  // Lower to prevent body warping

    const referenceImage = (
      typeof body.reference_image === 'string'
        ? body.reference_image
        : typeof body.referenceImage === 'string'
          ? body.referenceImage
          : undefined
    );

    // Use provided seed, or default from config (if >= 0), otherwise undefined (random)
    let seed = body.seed !== undefined ? String(body.seed) : undefined;
    if (seed === undefined && promptConfig.defaultSeed >= 0) {
      seed = String(promptConfig.defaultSeed);
    }

    if (!userPrompt && !promptConfig.masterPrompt) {
      return json(400, { error: 'Prompt required', detail: 'Provide a user prompt or configure a master prompt.' });
    }

    if (!referenceImage) {
      return json(400, { error: 'reference_image is required for inpainting' });
    }

    const facePadding = parseNumber(body.face_padding, promptConfig.facePadding);
    const controlnetEnabled = parseBool(body.controlnet_enabled ?? body.controlnetEnabled, promptConfig.controlnetEnabled);
    const controlnetType = body.controlnet_type ?? body.controlnetType ?? promptConfig.controlnetType;
    const controlnetScale = parseNumber(body.controlnet_scale ?? body.controlnetScale, promptConfig.controlnetScale);
    const faceidEnabled = parseBool(body.faceid_enabled ?? body.faceidEnabled, promptConfig.faceidEnabled);
    const faceidScale = parseNumber(body.faceid_scale ?? body.faceidScale, promptConfig.faceidScale);

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
        strength: strength,
        face_padding: facePadding,
        ip_adapter_scale: faceidEnabled ? faceidScale : ipAdapterScale,  // Use faceidScale if FaceID enabled
        use_controlnet: controlnetEnabled,
        controlnet_type: controlnetType,
        controlnet_scale: controlnetScale,
        use_faceid: faceidEnabled,
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
    const timeoutMs = parseNumber(body.timeoutMs, 180000);

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
        const image = ensureDataUrl(img);
        const maskImage = ensureDataUrl(out.mask_image || out.maskImage);
        const maskStartFraction = out.mask_start_fraction ?? out.maskStartFraction ?? null;
        const faceBoundingBox = out.face_bbox ?? out.faceBoundingBox ?? null;
        const segmentationInfo = out.segmentation ?? out.segmentationInfo ?? {};

        await appendLog(env, {
          timestamp: new Date().toISOString(),
          masterPrompt: promptConfig.masterPrompt,
          userPrompt,
          combinedPrompt: [promptConfig.masterPrompt, allowUserPrompt ? userPrompt : ''].filter(Boolean).join(', ').trim(),
          negativePrompt: negativePromptOverride,
          allowUserPrompt,
          width,
          height,
          steps,
          guidance,
          strength,
          ipAdapterScale,
          seed,
          referenceImage: ensureDataUrl(referenceImage),
          outputImage: image,
          maskImage,
          maskStartFraction,
          faceBoundingBox,
          segmentation: segmentationInfo,
          runpodOutput: out,
        });

        return json(200, {
          image,
          mask_image: out.mask_image,
          mask_start_fraction: maskStartFraction,
          face_bbox: faceBoundingBox,
          segmentation: segmentationInfo,
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
