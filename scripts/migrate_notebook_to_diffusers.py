"""
Migrate kaggle_ldm_stage2.ipynb to use Diffusers DDPMScheduler.

This script programmatically updates the notebook JSON to replace
the broken custom NoiseScheduler with proven Diffusers implementation.
"""

import json
import sys
from pathlib import Path


def find_and_replace_in_cell(cell, old_str, new_str):
    """Replace text in a cell's source."""
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if old_str in source:
            new_source = source.replace(old_str, new_str)
            # Split back into lines
            cell['source'] = new_source.splitlines(keepends=True)
            return True
    return False


def update_import_cell(cells):
    """Update the import cell to use Diffusers."""
    old_import = "from models.ldm.stage2.noise_scheduler import NoiseScheduler"
    new_import = "from diffusers import DDPMScheduler  # ✅ Using proven Diffusers implementation"
    
    for cell in cells:
        if find_and_replace_in_cell(cell, old_import, new_import):
            print("✅ Updated import cell")
            return True
    return False


def update_scheduler_init_cell(cells):
    """Update scheduler initialization cell."""
    old_code = '''# Initialize noise scheduler
print("\\nInitializing noise scheduler...")
scheduler = NoiseScheduler(num_timesteps=1000, schedule='cosine')
print(f"✅ Noise scheduler initialized (cosine, T=1000)")'''
    
    new_code = '''# Initialize noise scheduler (Diffusers DDPMScheduler)
print("\\nInitializing noise scheduler...")
scheduler = DDPMScheduler(
    num_train_timesteps=1000,
    beta_schedule="squaredcos_cap_v2",  # Improved cosine schedule
    prediction_type="epsilon",          # Predict noise (standard)
    clip_sample=False,                  # No hard clamping
    thresholding=False,                 # No dynamic thresholding
    variance_type="fixed_small",        # Conservative variance
    timestep_spacing="leading"          # Standard spacing
)
print(f"✅ Diffusers DDPMScheduler initialized")
print(f"   - num_train_timesteps: 1000")
print(f"   - beta_schedule: squaredcos_cap_v2")
print(f"   - prediction_type: epsilon")
print(f"   - variance_type: fixed_small")'''
    
    for cell in cells:
        if find_and_replace_in_cell(cell, old_code, new_code):
            print("✅ Updated scheduler initialization cell")
            return True
    return False


def remove_scheduler_to_device(cells):
    """Remove scheduler.to(device) call."""
    for cell in cells:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'scheduler.to(device)' in source:
                # Remove the line
                lines = cell['source']
                cell['source'] = [line for line in lines if 'scheduler.to(device)' not in line]
                print("✅ Removed scheduler.to(device) call")
                return True
    return False


def fix_add_noise_calls(cells):
    """Fix add_noise argument order in training loops."""
    old_pattern = "z_t = scheduler.add_noise(z_0, t, noise)"
    new_pattern = "z_t = scheduler.add_noise(z_0, noise, t)  # Diffusers: (original, noise, timesteps)"
    
    count = 0
    for cell in cells:
        if find_and_replace_in_cell(cell, old_pattern, new_pattern):
            count += 1
    
    if count > 0:
        print(f"✅ Fixed {count} add_noise calls")
    return count > 0


def update_sampling_function(cells):
    """Replace entire generate_samples function."""
    new_function = '''def generate_samples(unet, vq_model, scheduler, scaler, device, latent_shape, num_samples=8, num_steps=50):
    """
    Generate samples using Diffusers DDPMScheduler.
    
    Args:
        unet: Trained U-Net (will use EMA weights if available)
        vq_model: Frozen VQ decoder
        scheduler: Diffusers DDPMScheduler
        scaler: Latent scaler
        device: torch device
        latent_shape: Tuple (C, H, W) for latent dimensions
        num_samples: Number of samples to generate
        num_steps: Number of denoising steps (50 for respaced DDPM)
    
    Returns:
        Generated images tensor (num_samples, 3, 480, 640)
    """
    unet.eval()
    
    with torch.no_grad():
        # Configure scheduler for inference with respaced timesteps
        scheduler.set_timesteps(num_steps, device=device)
        
        # Start from pure noise
        z_t = torch.randn(num_samples, *latent_shape, device=device)
        
        # Iterative denoising using Diffusers scheduler
        for t in scheduler.timesteps:
            # Expand timestep to batch dimension
            t_tensor = t.unsqueeze(0).repeat(num_samples).to(device)
            
            # Predict noise
            noise_pred = unet(z_t, t_tensor)
            
            # Reverse diffusion step (handles all the math)
            result = scheduler.step(
                model_output=noise_pred,
                timestep=t,
                sample=z_t
            )
            
            z_t = result.prev_sample
        
        # z_t is now z_0 (denoised latent)
        z_0_scaled = z_t
        
        # Unscale
        z_0 = scaler.unscale(z_0_scaled)
        
        # Decode
        images = vq_model.decode(z_0)
        
        # Clamp to [0, 1]
        images = torch.clamp(images, 0, 1)
    
    unet.train()
    return images'''
    
    # Find the cell containing generate_samples
    for i, cell in enumerate(cells):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'def generate_samples(' in source and 'respaced DDPM sampling' in source:
                # Replace the entire cell
                cells[i]['source'] = new_function.splitlines(keepends=True)
                print("✅ Replaced generate_samples function")
                return True
    return False


def main():
    """Main migration function."""
    # Find notebook
    repo_root = Path(__file__).parent.parent
    notebook_path = repo_root / 'notebooks' / 'kaggle_ldm_stage2.ipynb'
    
    if not notebook_path.exists():
        print(f"❌ Notebook not found: {notebook_path}")
        return 1
    
    print(f"📖 Loading notebook: {notebook_path}")
    
    # Load notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    cells = notebook['cells']
    print(f"   Found {len(cells)} cells\n")
    
    # Apply migrations
    print("🔄 Applying Diffusers migration...\n")
    
    updates = [
        update_import_cell(cells),
        update_scheduler_init_cell(cells),
        remove_scheduler_to_device(cells),
        fix_add_noise_calls(cells),
        update_sampling_function(cells)
    ]
    
    if not all(updates):
        print("\n⚠️  Warning: Some updates may have failed")
        print("   Check the migration manually")
    
    # Create backup
    backup_path = notebook_path.with_suffix('.ipynb.backup')
    print(f"\n💾 Creating backup: {backup_path.name}")
    with open(backup_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    
    # Save updated notebook
    print(f"💾 Saving updated notebook: {notebook_path.name}")
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    
    print("\n✅ Migration complete!")
    print("\nNext steps:")
    print("  1. Review changes in Jupyter/Kaggle")
    print("  2. Run validation tests: python -m pytest tests/test_diffusers_integration.py")
    print("  3. Execute 5-epoch smoke test")
    print("  4. Compare samples with old broken implementation")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
