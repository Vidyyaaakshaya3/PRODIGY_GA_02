# Stable Diffusion - Final Fixed Version for Google Colab
# This version completely bypasses the KerasCV issues and uses Hugging Face Diffusers

# ==========================================
# SECTION 1: COMPLETE SOLUTION USING DIFFUSERS
# ==========================================

print("Installing Hugging Face Diffusers (more stable solution)...")

# Install the stable Diffusers library
!pip install diffusers transformers accelerate pillow matplotlib -q

import torch
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import time
import gc
import warnings
warnings.filterwarnings('ignore')

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

# ==========================================
# SECTION 2: MODEL INITIALIZATION
# ==========================================

def initialize_stable_diffusion():
    """Initialize Stable Diffusion using Hugging Face Diffusers"""
    print("Loading Stable Diffusion model...")

    try:
        # Load the model
        model_id = "runwayml/stable-diffusion-v1-5"

        if torch.cuda.is_available():
            print("Using GPU acceleration")
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16"
            )
            pipe = pipe.to("cuda")

            # Enable memory efficient attention
            pipe.enable_attention_slicing()
            pipe.enable_model_cpu_offload()

        else:
            print("Using CPU (will be slower)")
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
                use_safetensors=True
            )

        print("✅ Model loaded successfully!")
        return pipe

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Trying alternative model...")

        try:
            # Try with a different model
            model_id = "stabilityai/stable-diffusion-2-1-base"
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            if torch.cuda.is_available():
                pipe = pipe.to("cuda")

            print("✅ Alternative model loaded successfully!")
            return pipe

        except Exception as e2:
            print(f"❌ Alternative model also failed: {e2}")
            return None

# Initialize the model
pipe = initialize_stable_diffusion()

# ==========================================
# SECTION 3: IMAGE GENERATION FUNCTIONS
# ==========================================

def generate_image(prompt, negative_prompt="", num_steps=30, guidance_scale=7.5, width=512, height=512, seed=None):
    """Generate image using Stable Diffusion"""

    if pipe is None:
        print("❌ Model not loaded. Please restart and try again.")
        return None

    print(f"🎨 Generating: '{prompt}'")

    # Set seed for reproducibility
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    start_time = time.time()

    try:
        # Generate image
        with torch.autocast("cuda" if torch.cuda.is_available() else "cpu"):
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
            )

        image = result.images[0]
        generation_time = time.time() - start_time

        print(f"✅ Generated in {generation_time:.2f} seconds")
        return image

    except Exception as e:
        print(f"❌ Generation failed: {e}")
        return None

def display_and_save_image(image, prompt, save=True):
    """Display and optionally save the generated image"""

    if image is None:
        print("❌ No image to display")
        return

    # Display the image
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"Generated: {prompt}", fontsize=12, pad=20)
    plt.tight_layout()
    plt.show()

    if save:
        # Save the image
        timestamp = int(time.time())
        filename = f"generated_image_{timestamp}.png"
        image.save(filename)
        print(f"💾 Image saved as: {filename}")

# ==========================================
# SECTION 4: INTERACTIVE INPUT INTERFACE
# ==========================================

def interactive_generator():
    """Interactive interface for image generation"""

    if pipe is None:
        print("❌ Model not loaded. Cannot start interactive mode.")
        return

    print("\n" + "="*60)
    print("🎨 INTERACTIVE STABLE DIFFUSION GENERATOR")
    print("="*60)
    print("Enter your prompts below. Type 'quit' to exit.")
    print("You can also use advanced options!")
    print("-" * 60)

    while True:
        print("\n" + "🖼️  NEW IMAGE GENERATION")
        print("-" * 30)

        # Get prompt
        prompt = input("Enter your prompt: ").strip()

        if prompt.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break

        if not prompt:
            print("❌ Please enter a valid prompt.")
            continue

        # Ask for advanced options
        print("\n⚙️  Advanced Options (press Enter for defaults):")

        # Negative prompt
        negative_prompt = input("Negative prompt (what to avoid): ").strip()

        # Number of steps
        try:
            steps_input = input("Number of steps (default 30): ").strip()
            num_steps = int(steps_input) if steps_input else 30
            num_steps = max(10, min(100, num_steps))  # Clamp between 10-100
        except:
            num_steps = 30

        # Guidance scale
        try:
            guidance_input = input("Guidance scale (default 7.5): ").strip()
            guidance_scale = float(guidance_input) if guidance_input else 7.5
            guidance_scale = max(1.0, min(20.0, guidance_scale))  # Clamp between 1-20
        except:
            guidance_scale = 7.5

        # Seed
        try:
            seed_input = input("Seed for reproducibility (optional): ").strip()
            seed = int(seed_input) if seed_input else None
        except:
            seed = None

        # Generate image
        print(f"\n🚀 Generating image...")
        print(f"   Prompt: {prompt}")
        if negative_prompt:
            print(f"   Negative: {negative_prompt}")
        print(f"   Steps: {num_steps}, Guidance: {guidance_scale}")
        if seed:
            print(f"   Seed: {seed}")

        image = generate_image(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            seed=seed
        )

        if image:
            display_and_save_image(image, prompt)

            # Ask if user wants to continue
            continue_choice = input("\n🔄 Generate another image? (y/n): ").strip().lower()
            if continue_choice in ['n', 'no']:
                print("👋 Goodbye!")
                break
        else:
            print("❌ Generation failed. Please try again.")

# ==========================================
# SECTION 5: QUICK GENERATION FUNCTION
# ==========================================

def quick_generate(prompt, steps=30):
    """Quick generation function for simple use"""
    image = generate_image(prompt, num_steps=steps)
    if image:
        display_and_save_image(image, prompt)
    return image

# ==========================================
# SECTION 6: EXAMPLE GENERATIONS
# ==========================================

def run_examples():
    """Run example generations"""

    if pipe is None:
        print("❌ Model not loaded. Cannot run examples.")
        return

    print("\n" + "="*50)
    print("🎨 EXAMPLE GENERATIONS")
    print("="*50)

    examples = [
        {
            "prompt": "a cute cat sitting on a wooden table, photorealistic, detailed",
            "negative": "blurry, low quality, cartoon"
        },
        {
            "prompt": "a majestic mountain landscape at sunset, beautiful colors",
            "negative": "dark, gloomy, ugly"
        },
        {
            "prompt": "a robot making coffee in a modern kitchen, futuristic",
            "negative": "old, broken, messy"
        },
        {
            "prompt": "a magical forest with glowing mushrooms, fantasy art",
            "negative": "realistic, boring, plain"
        },
        {
            "prompt": "a red apple on a white background, simple, clean",
            "negative": "complex, cluttered, multiple objects"
        }
    ]

    for i, example in enumerate(examples, 1):
        print(f"\n--- Example {i}/5 ---")
        print(f"Prompt: {example['prompt']}")

        image = generate_image(
            prompt=example['prompt'],
            negative_prompt=example['negative'],
            num_steps=25
        )

        if image:
            display_and_save_image(image, f"Example {i}")
        else:
            print("❌ Example generation failed")

        # Small delay between generations
        time.sleep(1)

# ==========================================
# SECTION 7: INITIAL SETUP TEST
# ==========================================

if pipe is not None:
    print("\n" + "="*50)
    print("🧪 TESTING SETUP")
    print("="*50)

    # Test with a simple prompt
    test_prompt = "a simple red rose, beautiful, detailed"
    print(f"Testing with: '{test_prompt}'")

    test_image = generate_image(test_prompt, num_steps=20)

    if test_image:
        display_and_save_image(test_image, "Setup Test")
        print("\n✅ SETUP SUCCESSFUL!")
        print("\n🎉 Your Stable Diffusion is ready to use!")

        # Show usage options
        print("\n" + "="*50)
        print("📋 USAGE OPTIONS")
        print("="*50)
        print("1. Interactive Mode (Recommended):")
        print("   interactive_generator()")
        print("\n2. Quick Generation:")
        print("   quick_generate('your prompt here')")
        print("\n3. Advanced Generation:")
        print("   image = generate_image('prompt', negative_prompt='avoid this', num_steps=30)")
        print("   display_and_save_image(image, 'prompt')")
        print("\n4. Run Examples:")
        print("   run_examples()")

        # Automatically start interactive mode
        print("\n" + "="*50)
        print("🚀 STARTING INTERACTIVE MODE")
        print("="*50)

        interactive_generator()

    else:
        print("❌ Setup test failed")

else:
    print("\n" + "="*50)
    print("❌ SETUP FAILED")
    print("="*50)
    print("Please try the following:")
    print("1. Restart runtime: Runtime → Restart Runtime")
    print("2. Check GPU settings: Runtime → Change Runtime Type → GPU")
    print("3. Run this code again")
    print("4. If issues persist, try running in a new notebook")

# ==========================================
# SECTION 8: ADDITIONAL UTILITIES
# ==========================================

def batch_generate(prompts, negative_prompt="", num_steps=30):
    """Generate multiple images from a list of prompts"""

    if pipe is None:
        print("❌ Model not loaded")
        return

    results = []

    for i, prompt in enumerate(prompts, 1):
        print(f"\n🎨 Generating {i}/{len(prompts)}: {prompt}")

        image = generate_image(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_steps=num_steps
        )

        if image:
            display_and_save_image(image, f"Batch {i}: {prompt}")
            results.append(image)
        else:
            results.append(None)

    return results

def style_transfer_prompts():
    """Generate the same subject in different art styles"""

    if pipe is None:
        print("❌ Model not loaded")
        return

    subject = input("Enter a subject (e.g., 'a cat', 'a house', 'a person'): ").strip()

    if not subject:
        print("❌ Please enter a valid subject")
        return

    styles = [
        "photorealistic, detailed, high quality",
        "oil painting, classical art style",
        "watercolor painting, soft, artistic",
        "digital art, vibrant colors, modern",
        "sketch, pencil drawing, artistic",
        "anime style, manga, colorful",
        "cyberpunk style, neon, futuristic",
        "vintage, retro, old photograph style"
    ]

    print(f"\n🎨 Generating '{subject}' in different styles:")

    for i, style in enumerate(styles, 1):
        prompt = f"{subject}, {style}"
        print(f"\n--- Style {i}/8: {style} ---")

        image = generate_image(prompt, num_steps=25)
        if image:
            display_and_save_image(image, f"{subject} - {style}")

print("\n" + "="*60)
print("🎉 STABLE DIFFUSION SETUP COMPLETE!")
print("="*60)
print("Additional functions available:")
print("• batch_generate(['prompt1', 'prompt2', ...]) - Generate multiple images")
print("• style_transfer_prompts() - Same subject in different art styles")
print("• run_examples() - Run example generations")
print("• interactive_generator() - Start interactive mode again")
print("="*60)
