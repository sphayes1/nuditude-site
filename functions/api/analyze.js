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

    // Parse the uploaded image
    const formData = await request.formData();
    const imageFile = formData.get('image');

    if (!imageFile) {
      return new Response(JSON.stringify({ error: "No image file provided" }), {
        status: 400,
        headers: { "Content-Type": "application/json" }
      });
    }

    // Convert image to base64
    const arrayBuffer = await imageFile.arrayBuffer();
    const base64Image = btoa(String.fromCharCode(...new Uint8Array(arrayBuffer)));
    const mimeType = imageFile.type || 'image/jpeg';
    const dataUrl = `data:${mimeType};base64,${base64Image}`;

    // Step 1: Analyze image with GPT-4 Vision
    console.log("Analyzing image with GPT-4 Vision...");
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
    const description = visionData.choices[0].message.content;

    console.log("Image description:", description);

    // Step 2: Enhance description into stylized AI art prompt
    console.log("Enhancing prompt with GPT-4...");
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
            content: "You are an expert at writing AI image generation prompts. Convert factual descriptions into artistic, stylized prompts suitable for Stable Diffusion XL. Focus on artistic style, lighting, mood, and tasteful adult themes. Keep it tasteful and artistic, not explicit."
          },
          {
            role: "user",
            content: `Convert this description into a detailed AI art prompt for a tasteful, artistic boudoir-style image:\n\n${description}\n\nStyle: Professional photography, soft lighting, elegant, artistic, tasteful sensuality. Output ONLY the prompt, no explanation.`
          }
        ],
        max_tokens: 200,
        temperature: 0.8
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

    // Return the enhanced prompt
    return new Response(JSON.stringify({
      prompt: enhancedPrompt,
      originalDescription: description
    }), {
      status: 200,
      headers: { "Content-Type": "application/json" }
    });

  } catch (err) {
    console.error("Analyze error:", err);
    return new Response(JSON.stringify({
      error: "Server error",
      detail: String(err)
    }), {
      status: 500,
      headers: { "Content-Type": "application/json" }
    });
  }
}
