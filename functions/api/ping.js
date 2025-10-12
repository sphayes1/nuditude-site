// Cloudflare Pages Function: GET /api/ping
// Simple health check endpoint

export async function onRequest() {
  return new Response("ok", {
    status: 200,
    headers: { "Content-Type": "text/plain" }
  });
}
