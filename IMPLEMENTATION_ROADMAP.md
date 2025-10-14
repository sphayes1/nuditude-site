# Implementation Roadmap (Derived from `ai_outfit_swap_design.md`)

This roadmap breaks the high-level design into actionable milestones. Each milestone can be developed and deployed incrementally.

---

## Phase 1 — Enhanced Inference Pipeline (In Progress)
1. **Robust Mask Generation**
   - Integrate YOLOv8 garment detection (initially via pre-trained ONNX/PyTorch).
   - Use SAM (Segment Anything) to refine garment polygons.
   - Merge multiple garment segments into a binary mask; preserve head/face via InsightFace bbox exclusion.
   - Surface diagnostic overlays (mask preview) to frontend/admin.
2. **Manual Mask Editing**
   - Expose mask as editable layer (brush/eraser) in web UI.
   - Persist final mask in job request payload.
3. **ControlNet Support (Optional)**
   - Add OpenPose/Depth ControlNet to lock pose during inpainting.

## Phase 2 — Frontend Application (Next.js)
1. Scaffold Next.js + Tailwind project (migrate pages from static HTML).
2. Build core pages:
   - Upload & preview (with auto-mask overlay).
   - Prompt + style toggle panel.
   - Generation status & results gallery.
3. Implement mask editing (Konva.js/Fabric.js) & prompt presets.
4. Hook Next.js API routes / client calls to backend endpoints.

## Phase 3 — Backend Services (Beyond Cloudflare Functions)
1. Introduce Python API (FastAPI) for job management, authentication, and credit ledger.
2. Stand up PostgreSQL (job metadata, users, credit ledger) and Redis (queue/cache).
3. Add Celery/worker layer to orchestrate RunPod jobs and state transitions.
4. Move image storage to S3 with signed URL access; implement retention policies.

## Phase 4 — Monetization & Auth
1. User accounts with JWT sessions (email/password or OAuth).
2. Stripe integration:
   - Credit purchase endpoints & webhooks.
   - Credit reservation/refund logic around jobs.
3. Credit dashboard + history UI; include free-credit onboarding.

## Phase 5 — Safety & Compliance
1. Prompt moderation (keywords/LLM filter) + image moderation classifiers.
2. Consent & age verification flow (checkbox + policy links).
3. Implement DMCA workflow, audit logging, data delete tooling.
4. TLS/encryption policies and infrastructure hardening.

## Phase 6 — Deployment & Monitoring
1. Containerize services; deploy via AWS (EKS/ECS) with GPU inference nodes.
2. Configure Prometheus/Grafana, Sentry, structured logging.
3. Set up CI/CD (GitHub Actions) with automated tests, linting, and deploy pipelines.

## Phase 7 — Future Enhancements
- LoRA personalization (3–5 selfies).
- Mobile UI or native app packages.
- Community gallery with moderation tooling.
- Batch job support & multi-outfit try-on galleries.

---

### Immediate Next Milestone (Recommended)
- **Implement YOLOv8 + SAM clothing segmentation** feeding into the existing RunPod handler, with mask diagnostics surfaced in admin UI. This unlocks more precise clothing swaps before investing in frontend overhaul.

#### Segmentation Milestone — Task Breakdown
1. **Model Assets**
   - Add YOLOv8 garment detection weights (small model for speed).
   - Bundle SAM (ViT-B or smaller) checkpoint; manage caching/downloading similar to FaceID weights.
2. **Segmentation Service Module**
   - Create reusable Python module (e.g., `segmentation.py`) responsible for loading models and producing binary masks.
   - Accept PIL image, return binary/alpha mask plus polygon diagnostics.
3. **Handler Integration**
   - Call segmentation module before inpainting; combine segments, optionally expand/shrink mask margins.
   - Fall back to heuristic mask if models fail.
4. **Admin/Logging Updates**
   - Store segmentation metadata (model confidence, polygons) alongside existing logs.
   - Allow admin UI to display both auto mask and fallback state.
5. **Performance & Testing**
   - Benchmark model load time; consider persistent state to avoid reloading per job.
   - Add unit/integration test scripts verifying mask coverage on sample images.
