# Agentic Development Guide: AI Boudoir Generator Website

This guide is designed to be loaded into **Visual Studio Code** and used with **Agent Mode** (such as GitHub Copilot Agents or similar frameworks). It walks through the **entire lifecycle** of developing a privacy-first, AI-driven boudoir image generator website using a 3-stage pipeline:

**Analyze → Enhance → Generate.**

It includes setup, architecture, development milestones, and business-side details so an AI agent can execute tasks autonomously or under light human supervision.

---

## 🧭 Phase 0: Project Overview

**Goal:** Build a web app where users upload a selfie → AI analyzes → creates a stylized, fictionalized boudoir image → outputs a private, high-quality result.

**Core Value:** 100% consent-based, privacy-first, fictionalized image generation (NOT a deepfake or nudifier clone).

**Core Components:**
- Frontend: Next.js (React-based) + TailwindCSS + shadcn/ui
- Backend: FastAPI (Python) or Node.js (Express) + Redis Queue
- AI Inference: Stable Diffusion XL / 3.5, LLaVA-1.6, Mixtral-8x7B
- Storage: Temporary S3 (with auto-delete) or MinIO
- Auth & Payments: Stripe + CCBill/Segpay for adult-safe billing
- Deployment: Docker + Cloudflare + RunPod (GPU workers)

---

## ⚙️ Phase 1: Environment Setup

### 1.1 Local Setup
```bash
# Create project directories
mkdir ai-boudoir && cd ai-boudoir
mkdir backend frontend ai-pipeline docs

# Initialize repos
git init

# Python venv for backend
cd backend
python -m venv venv
source venv/bin/activate
pip install fastapi uvicorn redis aiohttp boto3

# Frontend
cd ../frontend
npx create-next-app@latest .
npm install tailwindcss @shadcn/ui axios react-dropzone stripe
```

### 1.2 GPU / AI Environment
```bash
# Create AI service env
cd ../ai-pipeline
python -m venv venv
source venv/bin/activate
pip install torch diffusers transformers accelerate safetensors bitsandbytes openai
```

### 1.3 .env Example
```
STRIPE_SECRET_KEY=sk_live_...
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
REDIS_URL=redis://localhost:6379
OPENAI_API_KEY=...
```

---

## 🧠 Phase 2: Core AI Pipeline

### 2.1 Step 1 – Analyze (Vision Description)
- Model: **LLaVA-1.6** or **IDEFICS**
- Input: User-uploaded selfie
- Output: Neutral, factual text description

```python
# ai_pipeline/analyze.py
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import torch

def analyze_image(image_path):
    model_id = "llava-hf/llava-1.6-vicuna-7b"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).eval()
    image = Image.open(image_path)
    prompt = "Describe this person’s appearance neutrally."
    inputs = processor(prompt, images=image, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=100)
    return processor.decode(output[0], skip_special_tokens=True)
```

### 2.2 Step 2 – Enhance (Prompt Enrichment)
- Model: **Mixtral 8x7B / Llama 3.1 70B**
- Task: Rewrite the neutral description into stylized, artistic prompts.

```python
# ai_pipeline/enhance.py
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def enhance_prompt(description, style="boudoir glamour"):
    prompt = f"Rewrite the following physical description into an artistic, sensual AI art prompt in the style of {style}. Include lighting, mood, and setting keywords. Ensure the tone is tasteful, adult, but not explicit.\n\n{description}"
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()
```

### 2.3 Step 3 – Generate (Image Creation)
- Model: **Stable Diffusion XL / 3.5**

```python
# ai_pipeline/generate.py
from diffusers import StableDiffusionXLPipeline
import torch

def generate_image(prompt, output_path):
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16
    ).to("cuda")

    image = pipe(prompt, guidance_scale=7.5, num_inference_steps=30).images[0]
    image.save(output_path)
    return output_path
```

### 2.4 Combined Flow
```python
# ai_pipeline/pipeline.py
from analyze import analyze_image
from enhance import enhance_prompt
from generate import generate_image


def run_pipeline(image_path):
    desc = analyze_image(image_path)
    styled_prompt = enhance_prompt(desc)
    output_path = f"outputs/result.png"
    generate_image(styled_prompt, output_path)
    return output_path
```

---

## 🌐 Phase 3: Backend API (FastAPI)

### 3.1 Endpoint Skeleton
```python
# backend/main.py
from fastapi import FastAPI, UploadFile
from ai_pipeline.pipeline import run_pipeline
import tempfile
import os

app = FastAPI()

@app.post("/generate")
async def generate(file: UploadFile):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    output = run_pipeline(tmp_path)
    os.remove(tmp_path)
    return {"result": output}
```

### 3.2 Privacy Middleware
- Auto-delete uploads after processing
- Log only minimal request metadata (no faces, no names)

### 3.3 Queuing
```python
# backend/worker.py
import redis
from rq import Queue
from pipeline import run_pipeline

r = redis.from_url(os.getenv("REDIS_URL"))
q = Queue("generation", connection=r)
```

---

## 🎨 Phase 4: Frontend (Next.js)

### 4.1 Pages & Components
- `/` → Landing Page + CTA (“Create My AI Photoshoot”)
- `/studio` → Upload form + style picker + progress loader
- `/result` → Display 4-up grid, purchase HD version

### 4.2 Upload & Preview Example
```tsx
// frontend/app/studio/page.tsx
import { useState } from "react";
import axios from "axios";

export default function StudioPage() {
  const [file, setFile] = useState<File | null>(null);
  const [image, setImage] = useState<string | null>(null);

  async function handleGenerate() {
    const formData = new FormData();
    if (!file) return;
    formData.append("file", file);
    const res = await axios.post("/api/generate", formData);
    setImage(res.data.result);
  }

  return (
    <div className="p-8 text-center">
      <h1 className="text-3xl font-bold">AI Boudoir Studio</h1>
      <input type="file" onChange={(e) => setFile(e.target.files?.[0] || null)} />
      <button onClick={handleGenerate} className="btn-primary mt-4">Generate</button>
      {image && <img src={image} className="mx-auto mt-8 rounded-2xl shadow" />}
    </div>
  );
}
```

---

## 💰 Phase 5: Monetization & Subscription Flow

### 5.1 Stripe Integration (Frontend)
```tsx
// frontend/lib/stripe.ts
import { loadStripe } from "@stripe/stripe-js";
export const stripePromise = loadStripe(process.env.NEXT_PUBLIC_STRIPE_PK!);
```

### 5.2 Backend Checkout Route
```python
# backend/payments.py
from fastapi import APIRouter
import stripe
import os

router = APIRouter()
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")

@router.post("/create-checkout-session")
def create_checkout():
    session = stripe.checkout.Session.create(
        payment_method_types=['card'],
        line_items=[{
            'price_data': {
                'currency': 'usd',
                'product_data': {'name': 'AI Boudoir Pro Subscription'},
                'unit_amount': 1200,
            },
            'quantity': 1,
        }],
        mode='subscription',
        success_url='https://yourdomain.com/success',
        cancel_url='https://yourdomain.com/cancel',
    )
    return {"url": session.url}
```

---

## 🔒 Phase 6: Legal, Safety, and Privacy

**Policies to include:**
- **Terms of Use:** user must own photo, 18+, no minors or celebrity likeness
- **Privacy:** uploaded photos deleted immediately; generated prompts anonymized
- **Data Retention:** output stored max 24 hours
- **Compliance:** age gate (checkbox), adult payment gateway (CCBill/Segpay)

**Security:**
- Run AI generation in isolated GPU worker containers
- Use presigned S3 URLs (auto-expire)
- Delete input files post-process

---

## 📈 Phase 7: Marketing & Growth

- Programmatic SEO: thousands of pages for styles, body types, keywords like “AI boudoir generator”, “AI cosplay photoshoot”
- Referral: 20% affiliate for creators
- Style Pack Drops weekly to retain subscribers
- Email drip automation for free-to-paid conversions

---

## 🧩 Phase 8: Expansion Roadmap

- Add **ControlNet Pose Picker**
- Add **Batch Boudoir Sets (8–24 poses)**
- Add **Animation Mode (short clips)**
- Add **Creator Dashboard** with analytics (downloads, styles used)

---

## ✅ Final Launch Checklist

- [ ] GPU worker tested (SDXL prompt → output < 20s)
- [ ] File auto-delete verified
- [ ] Stripe checkout live
- [ ] SEO sitemap generated
- [ ] Privacy & TOS pages published
- [ ] CCBill integration for adult billing
- [ ] Email onboarding automation live
- [ ] Cloudflare SSL + DDoS

---

## 🧠 Agent Task Summary (for VSCode Agent Mode)

| Phase | Agent Task | Tools |
|-------|-------------|-------|
| 1 | Setup project structure | shell, git |
| 2 | Implement AI pipeline | Python, diffusers, transformers |
| 3 | Build backend API | FastAPI |
| 4 | Build frontend UI | Next.js, Tailwind |
| 5 | Integrate payments | Stripe API |
| 6 | Add legal/privacy pages | Markdown to HTML |
| 7 | SEO automation | Node scripts / Python static gen |
| 8 | Deploy stack | Docker, RunPod, Cloudflare |

---

**Next Step (Agent Command)**
```bash
# Step 1: Bootstrap repository and initialize FastAPI + Next.js
agent run setup_project --repo ai-boudoir --frontend nextjs --backend fastapi --ai diffusers
```

