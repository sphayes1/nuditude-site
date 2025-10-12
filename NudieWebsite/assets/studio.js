document.addEventListener('DOMContentLoaded', () => {
  const btn = document.getElementById('go');
  const promptEl = document.getElementById('prompt');
  const statusEl = document.getElementById('status');
  const resultEl = document.getElementById('result');

  if (!btn || !promptEl || !statusEl || !resultEl) {
    console.error('Studio elements missing');
    return;
  }

  const errorEl = document.createElement('div');
  errorEl.className = 'note';
  errorEl.style.color = '#f87171';
  errorEl.style.marginTop = '8px';
  resultEl.parentElement.insertBefore(errorEl, resultEl);

  (async () => {
    try {
      const r = await fetch('/.netlify/functions/ping');
      const t = await r.text();
      if (r.ok && t.trim() === 'ok') statusEl.textContent = 'Generator ready';
    } catch {}
  })();

  btn.addEventListener('click', async () => {
    const prompt = (promptEl.value || '').trim();
    if (!prompt) { alert('Please enter a prompt.'); return; }
    statusEl.textContent = 'Generating... ~30Ã¢â‚¬â€œ60s';
    errorEl.textContent = '';
    btn.disabled = true;
    resultEl.innerHTML = '';
    try {
      const res = await fetch('/.netlify/functions/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt })
      });
      const text = await res.text();
      let data; try { data = JSON.parse(text); } catch { data = { error: 'Bad response', detail: text }; }
      if (!res.ok) {
        const detail = data && data.detail ? (typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail)) : '';
        const short = detail ? (detail.length > 220 ? detail.slice(0,220)+'â€¦' : detail) : '';
        throw new Error((data && data.error ? data.error : 'Generation failed') + (short ? ': '+ short : ''));
      }
      const img = document.createElement('img');
      img.src = data.image;
      img.alt = 'Generated image';
      img.style.maxWidth = '100%';
      img.style.borderRadius = '12px';
      img.style.boxShadow = '0 10px 40px rgba(0,0,0,.4)';
      resultEl.appendChild(img);
      statusEl.textContent = 'Done';
    } catch (e) {
      statusEl.textContent = 'Error generating image';
      errorEl.textContent = e && e.message ? e.message : 'Unknown error';
      console.error(e);
    } finally {
      btn.disabled = false;
    }
  });
});