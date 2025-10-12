// Cloudflare Pages Function: POST /api/generate
// Body: { prompt: string }
// Requires env: RUNPOD_API_KEY, RUNPOD_ENDPOINT_ID

export async function onRequestPost(context) {
  const { request, env } = context;

  try {
    const body = await request.json();
    const { prompt } = body;

    if (!prompt || typeof prompt !== "string") {
      return new Response(JSON.stringify({ error: "Missing prompt" }), {
        status: 400,
        headers: { "Content-Type": "application/json" }
      });
    }

    if (!env.RUNPOD_API_KEY) {
      return new Response(JSON.stringify({ error: "Missing RUNPOD_API_KEY" }), {
        status: 500,
        headers: { "Content-Type": "application/json" }
      });
    }

    if (!env.RUNPOD_ENDPOINT_ID) {
      return new Response(JSON.stringify({ error: "Missing RUNPOD_ENDPOINT_ID" }), {
        status: 500,
        headers: { "Content-Type": "application/json" }
      });
    }

    const RUNPOD_ENDPOINT = `https://api.runpod.ai/v2/${env.RUNPOD_ENDPOINT_ID}/run`;

    // Submit the job to RunPod
    console.log("Submitting job to RunPod...");
    const createRes = await fetch(RUNPOD_ENDPOINT, {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${env.RUNPOD_API_KEY}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        input: {
          prompt: prompt,
          negative_prompt: "ugly, deformed, blurry, low quality, distorted, nsfw explicit",
          width: 768,
          height: 1024,
          num_inference_steps: 30,
          guidance_scale: 7.5
        }
      })
    });

    if (!createRes.ok) {
      const text = await createRes.text();
      console.error("RunPod request failed:", text);
      return new Response(JSON.stringify({ error: "RunPod request failed", detail: text }), {
        status: createRes.status,
        headers: { "Content-Type": "application/json" }
      });
    }

    const createData = await createRes.json();
    const jobId = createData.id;

    if (!jobId) {
      return new Response(JSON.stringify({ error: "No job ID returned", detail: createData }), {
        status: 500,
        headers: { "Content-Type": "application/json" }
      });
    }

    console.log("Job submitted, ID:", jobId);

    // Poll for completion
    const STATUS_ENDPOINT = `https://api.runpod.ai/v2/${env.RUNPOD_ENDPOINT_ID}/status/${jobId}`;
    const started = Date.now();
    const timeoutMs = 120000; // 120s timeout
    let outputUrl = null;

    while (Date.now() - started < timeoutMs) {
      await new Promise(r => setTimeout(r, 2000)); // Poll every 2 seconds

      const statusRes = await fetch(STATUS_ENDPOINT, {
        headers: { "Authorization": `Bearer ${env.RUNPOD_API_KEY}` }
      });

      if (!statusRes.ok) {
        const text = await statusRes.text();
        console.error("Status check failed:", text);
        return new Response(JSON.stringify({ error: "Status check failed", detail: text }), {
          status: statusRes.status,
          headers: { "Content-Type": "application/json" }
        });
      }

      const statusData = await statusRes.json();
      console.log("Job status:", statusData.status);

      if (statusData.status === "COMPLETED") {
        // RunPod returns output in statusData.output
        const output = statusData.output;
        console.log("Raw output from RunPod:", JSON.stringify(output));

        // Handle different output formats
        if (typeof output === "string") {
          outputUrl = output;
        } else if (output && output.image) {
          outputUrl = output.image;
        } else if (output && output.images && Array.isArray(output.images)) {
          outputUrl = output.images[0];
        } else if (Array.isArray(output)) {
          outputUrl = output[0];
        }

        console.log("Extracted outputUrl:", outputUrl);

        if (outputUrl) {
          // Check if it's a base64 data URL or a regular URL
          if (outputUrl.startsWith('data:')) {
            console.log("Output is base64 data URL");
          } else if (outputUrl.startsWith('http')) {
            console.log("Output is HTTP URL");
          } else {
            console.log("Output format unclear:", outputUrl.substring(0, 50));
          }
          break;
        } else {
          return new Response(JSON.stringify({
            error: "Invalid output format",
            detail: output
          }), {
            status: 500,
            headers: { "Content-Type": "application/json" }
          });
        }
      }

      if (statusData.status === "FAILED") {
        return new Response(JSON.stringify({
          error: "Generation failed",
          detail: statusData.error || statusData
        }), {
          status: 500,
          headers: { "Content-Type": "application/json" }
        });
      }

      // Status could be: IN_QUEUE, IN_PROGRESS, COMPLETED, FAILED
    }

    if (!outputUrl) {
      return new Response(JSON.stringify({ error: "Generation timed out after 120s" }), {
        status: 504,
        headers: { "Content-Type": "application/json" }
      });
    }

    console.log("Generation complete, output URL:", outputUrl);

    // Return the image URL
    return new Response(JSON.stringify({ image: outputUrl }), {
      status: 200,
      headers: { "Content-Type": "application/json" }
    });

  } catch (err) {
    console.error("Generate error:", err);
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
