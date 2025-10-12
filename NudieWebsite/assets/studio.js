document.addEventListener(''DOMContentLoaded'', function () {
  var btn = document.getElementById(''go'');
  var promptEl = document.getElementById(''prompt'');
  var statusEl = document.getElementById(''status'');
  var resultEl = document.getElementById(''result'');

  if (!btn || !promptEl || !statusEl || !resultEl) {
    console.error(''Studio elements missing'');
    return;
  }

  var errorEl = document.createElement(''div'');
  errorEl.className = ''note'';
  errorEl.style.color = ''#f87171'';
  errorEl.style.marginTop = ''8px'';
  resultEl.parentElement.insertBefore(errorEl, resultEl);

  (async function () {
    try {
      var r = await fetch(''/api/ping'');
      var t = await r.text();
      if (r.ok && t.trim() === ''ok'') statusEl.textContent = ''Generator ready'';
    } catch (e) {}
  })();

  btn.addEventListener(''click'', async function () {
    var prompt = (promptEl.value || '''').trim();
    if (!prompt) { alert(''Please enter a prompt.''); return; }
    statusEl.textContent = ''Generating... ~30-60s'';
    errorEl.textContent = '';
    btn.disabled = true;
    resultEl.innerHTML = '';
    try {
      var res = await fetch(''/api/generate'', {
        method: ''POST'',
        headers: { ''Content-Type'': ''application/json'' },
        body: JSON.stringify({ prompt: prompt })
      });
      var text = await res.text();
      var data; try { data = JSON.parse(text); } catch (_) { data = { error: ''Bad response'', detail: text }; }
      if (!res.ok) {
        var detail = data && data.detail ? (typeof data.detail === ''string'' ? data.detail : JSON.stringify(data.detail)) : '';
        var short = detail ? (detail.length > 220 ? detail.slice(0,220)+''...'' : detail) : '';
        throw new Error((data && data.error ? data.error : ''Generation failed'') + (short ? '': '' + short : ''''));
      }
      var img = document.createElement(''img'');
      img.src = data.image;
      img.alt = ''Generated image'';
      img.style.maxWidth = ''100%'';
      img.style.borderRadius = ''12px'';
      img.style.boxShadow = ''0 10px 40px rgba(0,0,0,.4)'';
      resultEl.appendChild(img);
      statusEl.textContent = ''Done'';
    } catch (e) {
      statusEl.textContent = ''Error generating image'';
      errorEl.textContent = e && e.message ? e.message : ''Unknown error'';
      console.error(e);
    } finally {
      btn.disabled = false;
    }
  });
});