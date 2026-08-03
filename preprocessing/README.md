# Thermal Image Preprocessing Pipeline

Configurable preprocessing pipeline for thermal image mixed-pixel detection research.

## Overview

The preprocessing pipeline supports multiple optional operations without code modification. All operations are configured via `configs/preprocessing.yaml`.

## Features

- **Modular Design**: Each operation is independent and can be enabled/disabled
- **Configuration-Driven**: Change pipeline behavior by editing YAML only
- **Reproducible**: Default configuration reproduces paper requirements
- **Extensible**: Easy to add new preprocessing operations
- **Parallel Processing**: Multi-core support for efficient batch processing
- **Complete Traceability**: Metadata tracking for all operations

## Supported Operations

### 1. Resize ✓ (Required by Paper)

Resize all images to target resolution.

**Paper Requirement**: All images MUST be resized to 640×480 before training.

```yaml
resize:
  enabled: true
  width: 640
  height: 480
  preserve_aspect_ratio: false
  interpolation:
    downsample: "area"    # INTER_AREA (best for downsampling)
    upsample: "linear"    # INTER_LINEAR
```

**Interpolation Options**:
- `nearest`: Nearest neighbor (fastest, lowest quality)
- `linear`: Bilinear interpolation (good balance)
- `cubic`: Bicubic interpolation (slower, higher quality)
- `area`: Pixel area relation (best for downsampling)
- `lanczos`: Lanczos interpolation (highest quality, slowest)

### 2. Grayscale Conversion

Convert color images to single-channel grayscale.

```yaml
grayscale:
  enabled: false
```

Useful for models that operate on single-channel input.

### 3. Normalization

Normalize pixel intensities to standard ranges.

```yaml
normalization:
  enabled: false
  method: "minmax"      # Options: minmax, zscore, none
  min_value: 0.0        # For minmax: target minimum
  max_value: 1.0        # For minmax: target maximum
  per_image: true       # Use per-image or dataset statistics
```

**Methods**:
- `minmax`: Scale values to [min_value, max_value]
- `zscore`: Standardize to mean=0, std=1
- `none`: No normalization

**Use Cases**:
- `minmax` with [0, 1]: Common for neural networks
- `zscore`: Standard normalization for deep learning
- Dataset statistics: Set via `normalizer.set_dataset_statistics()`

### 4. Histogram Equalization

Global histogram equalization for contrast enhancement.

```yaml
histogram_equalization:
  enabled: false
```

**Behavior**:
- Grayscale images: Apply directly
- Color images: Convert to HSV, equalize V channel, convert back

**Note**: Do NOT use with CLAHE (choose one).

### 5. CLAHE

Contrast Limited Adaptive Histogram Equalization for local contrast enhancement.

```yaml
clahe:
  enabled: false
  clip_limit: 2.0
  tile_grid_size: [8, 8]
```

**Parameters**:
- `clip_limit`: Contrast limiting threshold (higher = more contrast)
- `tile_grid_size`: Size of tiles for local equalization [width, height]

**Behavior**:
- Grayscale images: Apply directly
- Color images: Convert to HSV, apply to V channel, convert back

**Recommendation**: CLAHE often works better than global histogram equalization for thermal images.

### 6. Padding

Add padding to images (useful with `preserve_aspect_ratio: true`).

```yaml
padding:
  enabled: false
  mode: "constant"     # Options: constant, edge, reflect, symmetric
  fill_value: 0        # For constant mode (0-255)
```

**Padding Modes**:
- `constant`: Fill with constant value
- `edge`: Replicate edge pixels
- `reflect`: Reflect pixels at border
- `symmetric`: Symmetric reflection

## Operation Order

Operations are applied in this fixed order:

1. **Resize** - Change image dimensions
2. **Grayscale** - Convert to single channel
3. **Histogram Equalization** - Global contrast
4. **CLAHE** - Local adaptive contrast
5. **Normalization** - Scale pixel values
6. **Padding** - Add borders if needed

**Rationale**:
- Resize first to standardize dimensions
- Contrast enhancement before normalization
- Normalization after all intensity operations
- Padding last to add borders

## Usage

### Basic Usage

```bash
# Run with default configuration (paper settings)
python preprocessing/preprocess.py --config configs/preprocessing.yaml

# Dry run (analyze without processing)
python preprocessing/preprocess.py --config configs/preprocessing.yaml --dry-run

# Custom log level
python preprocessing/preprocess.py --config configs/preprocessing.yaml --log-level DEBUG
```

### From Python

```python
from preprocessing import (
    ImageLoader,
    PreprocessingPipeline,
    ConfigurableImageProcessor,
    load_config
)

# Load configuration
config = load_config('configs/preprocessing.yaml')

# Initialize components
loader = ImageLoader()
pipeline = PreprocessingPipeline(config['preprocessing'])
processor = ConfigurableImageProcessor(loader, pipeline)

# Process single image
metadata = processor.process_image(
    input_path='path/to/input.jpg',
    output_path='path/to/output.jpg'
)

# Check enabled operations
enabled_ops = pipeline.get_enabled_operations()
print(f"Enabled operations: {enabled_ops}")
```

## Configuration Examples

### Example 1: Paper Default (Resize Only)

Reproduces paper requirements - resize to 640×480 only.

```yaml
preprocessing:
  resize:
    enabled: true
    width: 640
    height: 480
  grayscale:
    enabled: false
  normalization:
    enabled: false
  histogram_equalization:
    enabled: false
  clahe:
    enabled: false
  padding:
    enabled: false
```

### Example 2: CLAHE + Normalization

Apply CLAHE for contrast, then normalize to [0, 1].

```yaml
preprocessing:
  resize:
    enabled: true
    width: 640
    height: 480
  clahe:
    enabled: true
    clip_limit: 2.0
    tile_grid_size: [8, 8]
  normalization:
    enabled: true
    method: "minmax"
    min_value: 0.0
    max_value: 1.0
  grayscale:
    enabled: false
  histogram_equalization:
    enabled: false
  padding:
    enabled: false
```

### Example 3: Grayscale + Z-score

Convert to grayscale and standardize.

```yaml
preprocessing:
  resize:
    enabled: true
    width: 640
    height: 480
  grayscale:
    enabled: true
  normalization:
    enabled: true
    method: "zscore"
    per_image: true
  histogram_equalization:
    enabled: false
  clahe:
    enabled: false
  padding:
    enabled: false
```

See `configs/preprocessing_examples.yaml` for more examples.

## Output

### Processed Images

Processed images are saved to `datasets/processed/` with the same directory structure as input.

```
datasets/processed/
├── Chilli_Leaves/
│   ├── Bacterial Spot/
│   ├── Healthy Leaves/
│   └── ...
├── Okra/
│   └── Okra_Dataset/
│       ├── adequate_matured_Okra/
│       └── over_matured_Okra/
└── No_Mixed/
    └── No_mixed/
```

### Metadata

Complete metadata is saved to `datasets/metadata/preprocessing/`:

- `metadata.csv`: Per-image processing details
- `metadata.json`: Same data in JSON format
- `preprocessing_summary.md`: Human-readable summary
- `preprocessing_summary.json`: Machine-readable summary

**Metadata Fields**:
- Input/output paths
- Original/processed resolutions
- File sizes (before/after)
- Processing time
- Operations applied
- Operation-specific parameters

### Reports

Logs are saved to `reports/preprocessing/`:

- `preprocessing.log`: Detailed processing log

## Best Practices

### Thermal Images

1. **CLAHE over Histogram Equalization**: CLAHE preserves local details better
2. **Per-image Normalization**: Thermal images can have varying intensity ranges
3. **Preserve Original Data**: Keep raw datasets unchanged in `datasets/raw/`

### Model Training

1. **Normalization**: Apply normalization if your model expects normalized inputs
2. **Consistency**: Use the same preprocessing for training and inference
3. **Documentation**: Track which configuration was used for each experiment

### Performance

1. **Parallel Processing**: Adjust `max_workers` based on CPU cores
2. **Overwrite**: Set `overwrite: false` to skip already processed images
3. **Verify**: Enable `verify: true` to catch processing errors

## Extending the Pipeline

To add a new preprocessing operation:

1. Create a new module in `preprocessing/` (e.g., `my_operation.py`)
2. Implement a class with a processing method that returns `(image, metadata)`
3. Add the operation to `PreprocessingPipeline` in `pipeline.py`
4. Add configuration options to `configs/preprocessing.yaml`
5. Update `preprocessing/__init__.py` exports
6. Document the operation in this README

## Troubleshooting

### Images not being processed

- Check `overwrite: false` - already processed images are skipped
- Verify input paths in configuration
- Check file extensions in `valid_extensions`

### Wrong output format

- Float images from normalization are automatically converted to uint8
- Normalized [0, 1] range is scaled to [0, 255] for saving
- Use `verify: true` to check processing

### Memory issues

- Reduce `max_workers` for lower memory usage
- Process datasets individually
- Disable unnecessary operations

### Different results

- Ensure same configuration between runs
- Check `per_image` vs dataset statistics for normalization
- Verify operation order matches expectations

## References

- Original Paper: [Insert paper reference]
- OpenCV Documentation: https://docs.opencv.org/
- CLAHE: Zuiderveld, K. (1994). "Contrast Limited Adaptive Histogram Equalization"

## Version History

- **v2.0.0**: Configurable preprocessing pipeline
- **v1.1.0**: Deduplication module
- **v1.0.0**: Basic preprocessing (resize only)
