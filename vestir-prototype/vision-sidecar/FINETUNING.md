# Fine-tuning gate (DeepFashion2 / SigLIP / YOLO)

Do **not** start CLIP, SigLIP, or YOLO fine-tuning on closet data until you have:

- **At least 200–300 manually verified crops** from **real user wardrobes** (flat-lay, hanger, mirror selfies — not only editorial catalog shots).
- Ground-truth labels for **category** and **primary color** (and any attributes you care about).

**Why:** Models trained mainly on DeepFashion2 (or similar) excel on that distribution but often degrade on everyday closet photos. Use DF2 for **vocabulary and pre-training**, then **calibrate** on in-domain labeled data.

When you meet the gate, use scripts under `scripts/` and the reference repo at the workspace root (`Clothing-detection-and-attribute-identification-using-YoloV3-and-DeepFashion/`).
