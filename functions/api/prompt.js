const json = (status, data) =>
  new Response(JSON.stringify(data), {
    status,
    headers: {
      'Content-Type': 'application/json',
      'Cache-Control': 'no-store',
    },
  });

const PROMPT_KEY = 'master_prompt';
const NEGATIVE_KEY = 'negative_prompt';
const ALLOW_KEY = 'allow_user_prompt';
const STEPS_KEY = 'default_steps';
const GUIDANCE_KEY = 'default_guidance';
const STRENGTH_KEY = 'default_strength';
const SEED_KEY = 'default_seed';
const FACE_PADDING_KEY = 'face_padding';
const LOGS_KEY = 'generation_logs';

async function requireAuth(request, env) {
  const password = env.ADMIN_PASSWORD;
  if (!password) {
    return { ok: false, response: json(500, { error: 'Server misconfigured: ADMIN_PASSWORD missing.' }) };
  }

  const authHeader = request.headers.get('Authorization') || '';
  const token = authHeader.startsWith('Bearer ') ? authHeader.slice(7).trim() : '';

  if (!token || token !== password) {
    return { ok: false, response: json(401, { error: 'Unauthorized' }) };
  }

  return { ok: true };
}

async function loadPromptData(env) {
  if (!env.PROMPT_STORE) {
    throw new Error('PROMPT_STORE binding missing.');
  }
  const [masterPrompt, negativePrompt, allowValue, stepsValue, guidanceValue, strengthValue, seedValue, facePaddingValue, logsValue] = await Promise.all([
    env.PROMPT_STORE.get(PROMPT_KEY),
    env.PROMPT_STORE.get(NEGATIVE_KEY),
    env.PROMPT_STORE.get(ALLOW_KEY),
    env.PROMPT_STORE.get(STEPS_KEY),
    env.PROMPT_STORE.get(GUIDANCE_KEY),
    env.PROMPT_STORE.get(STRENGTH_KEY),
    env.PROMPT_STORE.get(SEED_KEY),
    env.PROMPT_STORE.get(FACE_PADDING_KEY),
    env.PROMPT_STORE.get(LOGS_KEY),
  ]);

  let logs = [];
  if (logsValue) {
    try {
      logs = JSON.parse(logsValue);
      if (!Array.isArray(logs)) {
        logs = [];
      }
    } catch (error) {
      logs = [];
    }
  }

  return {
    masterPrompt: masterPrompt || '',
    negativePrompt: negativePrompt || '',
    allowUserPrompt: allowValue === null ? false : allowValue === 'true',
    defaultSteps: stepsValue ? Number(stepsValue) : 28,
    defaultGuidance: guidanceValue ? Number(guidanceValue) : 5,
    defaultStrength: strengthValue ? Number(strengthValue) : 0.75,
    defaultSeed: seedValue ? Number(seedValue) : -1,
    facePadding: facePaddingValue ? Number(facePaddingValue) : 0.05,
    logs,
  };
}

async function savePromptData(env, data) {
  if (!env.PROMPT_STORE) {
    throw new Error('PROMPT_STORE binding missing.');
  }

  const { masterPrompt, negativePrompt, allowUserPrompt, defaultSteps, defaultGuidance, defaultStrength, defaultSeed, facePadding } = data;
  await Promise.all([
    env.PROMPT_STORE.put(PROMPT_KEY, masterPrompt || ''),
    env.PROMPT_STORE.put(NEGATIVE_KEY, negativePrompt || ''),
    env.PROMPT_STORE.put(ALLOW_KEY, allowUserPrompt ? 'true' : 'false'),
    env.PROMPT_STORE.put(STEPS_KEY, Number.isFinite(defaultSteps) ? String(defaultSteps) : '28'),
    env.PROMPT_STORE.put(GUIDANCE_KEY, Number.isFinite(defaultGuidance) ? String(defaultGuidance) : '5'),
    env.PROMPT_STORE.put(STRENGTH_KEY, Number.isFinite(defaultStrength) ? String(defaultStrength) : '0.75'),
    env.PROMPT_STORE.put(SEED_KEY, Number.isFinite(defaultSeed) ? String(defaultSeed) : '-1'),
    env.PROMPT_STORE.put(FACE_PADDING_KEY, Number.isFinite(facePadding) ? String(facePadding) : '0.05'),
  ]);
}

export async function onRequestGet(context) {
  try {
    const auth = await requireAuth(context.request, context.env);
    if (!auth.ok) return auth.response;

    const payload = await loadPromptData(context.env);
    return json(200, payload);
  } catch (error) {
    return json(500, { error: 'Failed to load prompt', detail: String(error) });
  }
}

export async function onRequestPost(context) {
  try {
    const auth = await requireAuth(context.request, context.env);
    if (!auth.ok) return auth.response;

    const body = await context.request.json().catch(() => ({}));
    const parseBool = (value) => {
      if (typeof value === 'boolean') return value;
      if (typeof value === 'number') return value !== 0;
      if (typeof value === 'string') {
        const normalized = value.trim().toLowerCase();
        if (['true', '1', 'yes', 'on'].includes(normalized)) return true;
        if (['false', '0', 'no', 'off'].includes(normalized)) return false;
      }
      return false;
    };

    const masterPrompt = typeof body.masterPrompt === 'string' ? body.masterPrompt.trim() : '';
    const negativePrompt = typeof body.negativePrompt === 'string' ? body.negativePrompt.trim() : '';
    const allowUserPrompt = parseBool(body.allowUserPrompt);
    const defaultSteps = Number(body.defaultSteps);
    const defaultGuidance = Number(body.defaultGuidance);
    const defaultStrength = Number(body.defaultStrength);
    const defaultSeed = Number(body.defaultSeed);
    const facePadding = Number(body.facePadding);

    await savePromptData(context.env, {
      masterPrompt,
      negativePrompt,
      allowUserPrompt,
      defaultSteps: Number.isFinite(defaultSteps) ? defaultSteps : 28,
      defaultGuidance: Number.isFinite(defaultGuidance) ? defaultGuidance : 5,
      defaultStrength: Number.isFinite(defaultStrength) ? defaultStrength : 0.75,
      defaultSeed: Number.isFinite(defaultSeed) ? defaultSeed : -1,
      facePadding: Number.isFinite(facePadding) ? facePadding : 0.05,
    });

    const payload = await loadPromptData(context.env);
    return json(200, { ok: true, config: payload });
  } catch (error) {
    return json(500, { error: 'Failed to save prompt', detail: String(error) });
  }
}
