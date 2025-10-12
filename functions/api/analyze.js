// Cloudflare Pages Function: POST /api/analyze
// Body: FormData with 'image' file
// Requires env: OPENAI_API_KEY
// Returns: { prompt: string }

export async function onRequestPost(context) {
  const { request, env } = context;

  try {
    // Check for OpenAI API key
    if (!env.OPENAI_API_KEY) {
      return new Response(JSON.stringify({ error: "Missing OPENAI_API_KEY" }), {
        status: 500,
        headers: { "Content-Type": "application/json" }
      });
    }

    // Parse the uploaded image and spice level
    const formData = await request.formData();
    const imageFile = formData.get('image');
    const spiceLevel = parseInt(formData.get('spiceLevel') || '1');
    const baseDescription = formData.get('baseDescription'); // Reuse cached description

    console.log("Spice level requested:", spiceLevel);
    console.log("Base description provided:", !!baseDescription);

    let description = baseDescription;

    // Only analyze image with Vision API if we don't have a base description
    if (!description && imageFile) {
      console.log("No base description - analyzing image with GPT-4 Vision...");

      // Convert image to base64 (Cloudflare Workers compatible method)
      const arrayBuffer = await imageFile.arrayBuffer();
      const bytes = new Uint8Array(arrayBuffer);

      // Convert bytes to base64 string
      let binary = '';
      const chunkSize = 0x8000; // 32KB chunks to avoid stack overflow
      for (let i = 0; i < bytes.length; i += chunkSize) {
        const chunk = bytes.subarray(i, Math.min(i + chunkSize, bytes.length));
        binary += String.fromCharCode.apply(null, chunk);
      }
      const base64Image = btoa(binary);

      const mimeType = imageFile.type || 'image/jpeg';
      const dataUrl = `data:${mimeType};base64,${base64Image}`;

      // Step 1: Analyze image with GPT-4 Vision
      const visionResponse = await fetch("https://api.openai.com/v1/chat/completions", {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${env.OPENAI_API_KEY}`,
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          model: "gpt-4o",
          messages: [
            {
              role: "user",
              content: [
                {
                  type: "text",
                  text: "Describe this person's appearance in detail. Focus on: physical features, clothing style, pose, lighting, and setting. Be factual and neutral. Do not include names or identify individuals."
                },
                {
                  type: "image_url",
                  image_url: {
                    url: dataUrl
                  }
                }
              ]
            }
          ],
          max_tokens: 300
        })
      });

      if (!visionResponse.ok) {
        const errorText = await visionResponse.text();
        return new Response(JSON.stringify({ error: "Vision API failed", detail: errorText }), {
          status: visionResponse.status,
          headers: { "Content-Type": "application/json" }
        });
      }

      const visionData = await visionResponse.json();
      description = visionData.choices[0].message.content;

      console.log("Image description:", description);
    } else if (description) {
      console.log("Using cached base description to maintain likeness");
    } else {
      return new Response(JSON.stringify({ error: "No image or base description provided" }), {
        status: 400,
        headers: { "Content-Type": "application/json" }
      });
    }

    // Step 2: Enhance description into stylized AI art prompt with progressive spice
    console.log("Enhancing prompt with GPT-4...");

    // Define spice level instructions
    const spiceInstructions = {
      1: "Keep the person mostly clothed but in a more relaxed, casual pose. Remove outer layers like jackets or sweaters. Suggest elegant, artistic lighting. Tasteful and sophisticated.",
      2: "The person should be in lingerie or underwear. Elegant, high-end boudoir photography style. Soft lighting, artistic poses. Tasteful sensuality.",
      3: "Artistic nude photography. Implied nudity with strategic positioning, shadows, or partial coverage. Professional, tasteful, gallery-quality artistic photography. No explicit content.",
      4: "Full artistic nude in an elegant, sensual pose. High-end fashion photography aesthetic. Dramatic lighting, sophisticated composition. Artistic and tasteful, never pornographic."
    };

    const currentInstruction = spiceInstructions[Math.min(spiceLevel, 4)] || spiceInstructions[1];

    const enhanceResponse = await fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${env.OPENAI_API_KEY}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        model: "gpt-4o-mini",
        messages: [
          {
            role: "system",
            content: "You are an expert at writing AI image generation prompts for artistic boudoir photography. Convert factual descriptions into artistic, stylized prompts suitable for Stable Diffusion XL. Focus on artistic style, lighting, mood, and composition. Always maintain artistic taste and sophistication."
          },
          {
            role: "user",
            content: `Based on this person's description:\n${description}\n\nCreate a detailed Stable Diffusion prompt following these guidelines:\n${currentInstruction}\n\nInclude: artistic style, lighting (soft, rim, dramatic), mood, composition, camera angle. Use photography terms like 'bokeh', 'shallow depth of field', 'cinematic lighting'.\n\nOutput ONLY the prompt, no explanation or preamble.`
          }
        ],
        max_tokens: 250,
        temperature: 0.85
      })
    });

    if (!enhanceResponse.ok) {
      const errorText = await enhanceResponse.text();
      return new Response(JSON.stringify({ error: "Enhancement API failed", detail: errorText }), {
        status: enhanceResponse.status,
        headers: { "Content-Type": "application/json" }
      });
    }

    const enhanceData = await enhanceResponse.json();
    const enhancedPrompt = enhanceData.choices[0].message.content.trim();

    console.log("Enhanced prompt:", enhancedPrompt);

    // Return the enhanced prompt with spice level
    return new Response(JSON.stringify({
      prompt: enhancedPrompt,
      originalDescription: description,
      spiceLevel: spiceLevel,
      nextSpiceLevel: Math.min(spiceLevel + 1, 4),
      maxLevel: spiceLevel >= 4
    }), {
      status: 200,
      headers: { "Content-Type": "application/json" }
    });

  } catch (err) {
    console.error("Analyze error:", err);
    return new Response(JSON.stringify({
      error: "Server error",
      detail: err.message || String(err),
      stack: err.stack
    }), {
      status: 500,
      headers: { "Content-Type": "application/json" }
    });
  }
}
