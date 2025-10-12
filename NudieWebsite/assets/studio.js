document.addEventListener('DOMContentLoaded', () => {
  // Elements
  const imageUpload = document.getElementById('imageUpload');
  const preview = document.getElementById('preview');
  const analyzeBtn = document.getElementById('analyzeBtn');
  const promptEl = document.getElementById('prompt');
  const generateBtn = document.getElementById('generateBtn');
  const statusEl = document.getElementById('status');
  const resultEl = document.getElementById('result');

  let selectedFile = null;

  // Health check
  (async () => {
    try {
      const r = await fetch('/api/ping');
      const t = await r.text();
      if (r.ok && t.trim() === 'ok') {
        console.log('Generator ready');
      }
    } catch (e) {
      console.error('Health check failed:', e);
    }
  })();

  // Handle file selection
  imageUpload.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) {
      selectedFile = null;
      analyzeBtn.disabled = true;
      preview.innerHTML = '';
      return;
    }

    // Validate file type
    if (!file.type.match('image/(jpeg|png|webp)')) {
      alert('Please upload a JPG, PNG, or WebP image');
      imageUpload.value = '';
      return;
    }

    // Validate file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
      alert('Image must be under 10MB');
      imageUpload.value = '';
      return;
    }

    selectedFile = file;
    analyzeBtn.disabled = false;

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
      preview.innerHTML = `<img src="${e.target.result}" style="max-width:300px;max-height:300px;border-radius:8px;box-shadow:0 4px 12px rgba(0,0,0,0.3);" alt="Preview" />`;
    };
    reader.readAsDataURL(file);
  });

  // Handle analyze & generate button
  analyzeBtn.addEventListener('click', async () => {
    if (!selectedFile) {
      alert('Please select an image first');
      return;
    }

    analyzeBtn.disabled = true;
    generateBtn.disabled = true;
    resultEl.innerHTML = '';
    statusEl.textContent = 'Step 1/2: Analyzing image with AI... ~10s';

    try {
      // Step 1: Analyze the image
      const formData = new FormData();
      formData.append('image', selectedFile);

      const analyzeRes = await fetch('/api/analyze', {
        method: 'POST',
        body: formData
      });

      const analyzeData = await analyzeRes.json();

      if (!analyzeRes.ok) {
        const errorMsg = analyzeData.error || 'Analysis failed';
        const detail = analyzeData.detail ? ` - ${analyzeData.detail}` : '';
        console.error('Analysis error:', analyzeData);
        throw new Error(errorMsg + detail);
      }

      const enhancedPrompt = analyzeData.prompt;
      console.log('Enhanced prompt:', enhancedPrompt);

      // Show the prompt in the textarea for user to see/edit
      promptEl.value = enhancedPrompt;

      // Step 2: Generate image from the prompt
      statusEl.textContent = 'Step 2/2: Generating new image... ~30-60s';

      const generateRes = await fetch('/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: enhancedPrompt })
      });

      const generateData = await generateRes.json();

      if (!generateRes.ok) {
        throw new Error(generateData.error || 'Generation failed');
      }

      // Display result
      const img = document.createElement('img');
      img.src = generateData.image;
      img.alt = 'Generated image';
      img.style.maxWidth = '100%';
      img.style.borderRadius = '12px';
      img.style.boxShadow = '0 10px 40px rgba(0,0,0,.4)';
      resultEl.appendChild(img);

      statusEl.textContent = '✅ Done! Generated from your uploaded image.';
      statusEl.style.color = '#10b981';

    } catch (e) {
      statusEl.textContent = '❌ Error: ' + (e.message || 'Unknown error');
      statusEl.style.color = '#ef4444';
      console.error(e);
    } finally {
      analyzeBtn.disabled = false;
      generateBtn.disabled = false;
    }
  });

  // Handle generate from prompt button
  generateBtn.addEventListener('click', async () => {
    const prompt = (promptEl.value || '').trim();

    if (!prompt) {
      alert('Please enter a prompt');
      return;
    }

    generateBtn.disabled = true;
    analyzeBtn.disabled = true;
    resultEl.innerHTML = '';
    statusEl.textContent = 'Generating from prompt... ~30-60s';
    statusEl.style.color = '';

    try {
      const generateRes = await fetch('/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt })
      });

      const generateData = await generateRes.json();

      if (!generateRes.ok) {
        throw new Error(generateData.error || 'Generation failed');
      }

      // Display result
      const img = document.createElement('img');
      img.src = generateData.image;
      img.alt = 'Generated image';
      img.style.maxWidth = '100%';
      img.style.borderRadius = '12px';
      img.style.boxShadow = '0 10px 40px rgba(0,0,0,.4)';
      resultEl.appendChild(img);

      statusEl.textContent = '✅ Done!';
      statusEl.style.color = '#10b981';

    } catch (e) {
      statusEl.textContent = '❌ Error: ' + (e.message || 'Unknown error');
      statusEl.style.color = '#ef4444';
      console.error(e);
    } finally {
      generateBtn.disabled = false;
      analyzeBtn.disabled = false;
    }
  });
});
