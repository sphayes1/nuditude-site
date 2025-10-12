# AI Boudoir – Static Site Starter

This is a static, privacy‑first landing site with legal pages and a monetization CTA. Replace placeholder links and deploy on any static host.

## Customize

- `index.html`: Update brand name, copy, and the checkout link under the "Buy" section. Use your Gumroad or Stripe hosted payment page.
- `privacy.html` / `terms.html`: Adjust language for your business.
- `contact.html`: Connect to Formspree, Netlify Forms, or your backend by setting the form `action` URL and enabling the button.

## Security & Privacy

- CSP is set via `<meta http-equiv="Content-Security-Policy">` for strict, self‑hosted assets only.
- No external scripts by default; keep it that way for safety and speed.
- Do not store uploads on this static site. If you add AI features, process files server‑side and auto‑delete.

## Hosting

### Netlify
1. Create a new site from this folder.
2. Drag‑and‑drop in the Netlify UI or connect a repo.

### Vercel
1. Create New Project > Framework: "Other".
2. Set output directory to the project root.

### Cloudflare Pages
1. Create a project and point it at the repo.
2. Build command: none. Output directory: `/`.

### GitHub Pages
1. Push this folder to a repo.
2. Settings > Pages > Deploy from branch > `main` and root directory.

## Monetization Ideas

- Sell the guide via Gumroad/Stripe hosted checkout (fastest). 
- Add a tip jar (Buy Me a Coffee, Ko‑fi) via an external link.
- Collect emails for future product drops (Beehiiv/Substack hosted forms).

## Next Steps

- Swap the placeholder checkout URL with your real link.
- Add a brand logo and images (local files to keep CSP simple).
- If you later build an AI studio, host it separately behind proper age gate, consent flows, and adult‑compliant processors.

