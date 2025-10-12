document.addEventListener('DOMContentLoaded', () => {
  // Elements
  const imageUpload = document.getElementById('imageUpload');
  const preview = document.getElementById('preview');
  const generateBtn = document.getElementById('generateBtn');
  const extraSpicyBtn = document.getElementById('extraSpicyBtn');
  const nextLevelBtn = document.getElementById('nextLevelBtn');
  const statusEl = document.getElementById('status');
  const resultEl = document.getElementById('result');
  const promptDebug = document.getElementById('promptDebug');
  const promptText = document.getElementById('promptText');

  let selectedFile = null;
  let currentSpiceLevel = 1;
  let lastGeneratedImage = null;
  let baseDescription = null; // Store the original description to maintain likeness
  let detectedStartLevel = 1; // Smart detection of starting clothing level

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
      generateBtn.disabled = true;
      nextLevelBtn.style.display = 'none';
      preview.innerHTML = '';
      resultEl.innerHTML = '';
      currentSpiceLevel = 1;
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
    generateBtn.disabled = false;
    extraSpicyBtn.disabled = false;
    extraSpicyBtn.style.display = 'inline-block';
    currentSpiceLevel = 1;
    baseDescription = null; // Reset description for new image
    detectedStartLevel = 1;
    nextLevelBtn.style.display = 'none';
    resultEl.innerHTML = '';

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
      preview.innerHTML = `<img src="${e.target.result}" style="max-width:300px;max-height:300px;border-radius:8px;box-shadow:0 4px 12px rgba(0,0,0,0.3);" alt="Preview" />`;
    };
    reader.readAsDataURL(file);
  });

  // Main generate function
  async function generateImage(spiceLevel) {
    if (!selectedFile) {
      alert('Please select an image first');
      return;
    }

    generateBtn.disabled = true;
    extraSpicyBtn.disabled = true;
    nextLevelBtn.disabled = true;
    resultEl.innerHTML = '';

    const isExtraSpicy = (spiceLevel === 4 && currentSpiceLevel === 1);
    if (isExtraSpicy) {
      statusEl.textContent = '🔥🔥🔥 Extra Spicy mode! Going straight to maximum... ~30-60s';
    } else {
      statusEl.textContent = '🎨 Creating your artistic transformation... ~30-60s';
    }
    statusEl.style.color = '';

    try {
      // Step 1: Analyze the image with spice level
      const formData = new FormData();

      // Only send image on first generation (level 1), then reuse description
      if (spiceLevel === 1 || !baseDescription) {
        formData.append('image', selectedFile);
      }

      formData.append('spiceLevel', spiceLevel);

      // If we have a base description, send it to maintain likeness
      if (baseDescription && spiceLevel > 1) {
        formData.append('baseDescription', baseDescription);
      }

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
      console.log('Spice level:', analyzeData.spiceLevel);

      // Store the base description from first generation
      if (spiceLevel === 1 && analyzeData.originalDescription) {
        baseDescription = analyzeData.originalDescription;
        console.log('Stored base description for likeness preservation');
      }

      // Show prompt in debug section (hidden by default)
      if (promptDebug && promptText) {
        promptText.textContent = enhancedPrompt;
      }

      // Step 2: Generate image from the prompt
      statusEl.textContent = '🔥 Generating your image... ~30-60s';

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
      img.style.maxHeight = '600px';
      img.style.borderRadius = '12px';
      img.style.boxShadow = '0 10px 40px rgba(0,0,0,.4)';
      resultEl.appendChild(img);

      lastGeneratedImage = generateData.image;

      // Update spice level and show "Make it Spicier" button
      currentSpiceLevel = analyzeData.nextSpiceLevel;

      if (analyzeData.maxLevel) {
        statusEl.textContent = '🔥💯 Maximum spice level reached!';
        statusEl.style.color = '#ef4444';
        nextLevelBtn.style.display = 'none';
        extraSpicyBtn.style.display = 'none';
      } else {
        statusEl.textContent = '✅ Done! Want to make it even spicier?';
        statusEl.style.color = '#10b981';
        nextLevelBtn.style.display = 'inline-block';
        nextLevelBtn.disabled = false;
        extraSpicyBtn.style.display = 'none'; // Hide after first generation
      }

    } catch (e) {
      statusEl.textContent = '❌ Error: ' + (e.message || 'Unknown error');
      statusEl.style.color = '#ef4444';
      console.error(e);
    } finally {
      generateBtn.disabled = false;
    }
  }

  // Handle initial generate button
  generateBtn.addEventListener('click', () => {
    generateImage(1); // Start at spice level 1
  });

  // Handle "Extra Spicy" button (premium - skip straight to level 4)
  extraSpicyBtn.addEventListener('click', () => {
    if (confirm('Extra Spicy costs 2 credits and skips straight to maximum spice level. Continue?')) {
      generateImage(4); // Go directly to level 4
    }
  });

  // Handle "Next Level" button
  nextLevelBtn.addEventListener('click', () => {
    generateImage(currentSpiceLevel);
  });
});
