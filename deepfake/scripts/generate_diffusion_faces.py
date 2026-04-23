import argparse
import os
import random
import time

PROMPT_TEMPLATES = [
    "a photo of a {age} {ethnicity} {gender}, {expression}, {lighting}, {setting}, 85mm portrait",
    "headshot of a {age} {ethnicity} {gender}, {expression}, {lighting}",
    "realistic portrait of a {age} {ethnicity} {gender}, {expression}, {setting}",
    "candid photo of a {age} {ethnicity} {gender}, {lighting}, {setting}",
    "professional headshot, {age} {ethnicity} {gender}, {expression}",
]

AGES = ["young", "middle-aged", "elderly", "teenage", "20s", "30s", "40s", "50s"]
GENDERS = ["man", "woman"]
ETHNICITIES = [
    "asian", "european", "african", "hispanic", "middle eastern",
    "south asian", "east asian", "caucasian",
]
EXPRESSIONS = [
    "smiling", "serious", "looking at camera", "laughing",
    "thoughtful", "neutral expression", "slight smile",
]
LIGHTINGS = [
    "soft studio lighting", "natural daylight", "golden hour",
    "indoor warm light", "cinematic lighting", "window light",
]
SETTINGS = [
    "in an office", "outdoors park", "cafe background",
    "plain backdrop", "urban street", "living room",
]
NEGATIVE = (
    "blurry, low quality, deformed face, extra fingers, text, watermark, "
    "cartoon, anime, painting, sketch, ugly, asymmetric eyes"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate diffusion face images locally with Stable Diffusion XL"
    )
    parser.add_argument("--out_dir", type=str, default="data/extra/diffusion")
    parser.add_argument("--count", type=int, default=2000)
    parser.add_argument("--model_id", type=str,
                        default="stabilityai/stable-diffusion-xl-base-1.0",
                        help="HuggingFace model id. Smaller alt: "
                             "'runwayml/stable-diffusion-v1-5'")
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--guidance", type=float, default=6.0)
    parser.add_argument("--image_size", type=int, default=1024,
                        help="1024 for SDXL, 512 for SD 1.5")
    parser.add_argument("--batch", type=int, default=1,
                        help="Images per forward pass")
    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Starting filename index (for resuming)")
    return parser.parse_args()


def build_pipeline(model_id):
    import torch
    from diffusers import AutoPipelineForText2Image
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    pipe = AutoPipelineForText2Image.from_pretrained(
        model_id, torch_dtype=dtype, variant="fp16" if dtype == torch.float16 else None,
    )
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def random_prompt(rng):
    template = rng.choice(PROMPT_TEMPLATES)
    return template.format(
        age=rng.choice(AGES),
        ethnicity=rng.choice(ETHNICITIES),
        gender=rng.choice(GENDERS),
        expression=rng.choice(EXPRESSIONS),
        lighting=rng.choice(LIGHTINGS),
        setting=rng.choice(SETTINGS),
    )


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    import torch
    pipe = build_pipeline(args.model_id)
    rng = random.Random(args.seed_start)

    generated = 0
    skipped = 0
    start = time.time()

    while generated + skipped < args.count:
        idx = args.start_idx + generated + skipped
        path = os.path.join(args.out_dir, f"diffusion_{idx:06d}.jpg")
        if os.path.exists(path):
            skipped += 1
            continue
        prompt = random_prompt(rng)
        seed = args.seed_start + idx
        generator = torch.Generator(device=pipe.device).manual_seed(seed)
        try:
            image = pipe(
                prompt=prompt,
                negative_prompt=NEGATIVE,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                height=args.image_size,
                width=args.image_size,
                generator=generator,
            ).images[0]
        except Exception as e:
            print(f"  [{idx}] FAIL {e}", flush=True)
            continue
        image.save(path, quality=92)
        generated += 1
        if generated % 20 == 0:
            elapsed = time.time() - start
            rate = generated / max(elapsed, 1)
            eta = (args.count - generated - skipped) / max(rate, 1e-6)
            print(f"  [{idx}] generated={generated} skipped={skipped} "
                  f"rate={rate:.2f}/s eta={eta/60:.1f}min | prompt='{prompt[:60]}...'",
                  flush=True)

    total = time.time() - start
    print(f"\nDone. generated={generated} skipped={skipped} total={total:.0f}s")


if __name__ == "__main__":
    main()
