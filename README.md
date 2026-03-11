# WaveletStyleTransfer

Neural Style Transfer using **2D Haar Wavelet pooling** and **Whitening & Coloring Transform (WCT)**.  
Instead of discarding high-frequency spatial information via max/average pooling, the encoder decomposes each feature map into four Haar wavelet sub-bands (LL, LH, HL, HH). The high-frequency bands are preserved as skip connections and reused during decoding, resulting in stylized images with finer structural detail.

> `utils/core.py` and `utils/io.py` are adapted from [NVIDIA FastPhotoStyle](https://github.com/NVIDIA/FastPhotoStyle) under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

---

## How It Works

```
Content Image ──┐
                ▼
         Encoder (VGG-like)
         conv → HaarPool2D × 3   ← splits feature maps into LL / LH / HL / HH
                ▼
         WCT (feature whitening + style recoloring via SVD)
                ▼
         Decoder (mirror)
         HaarUnpool2D × 3        ← fuses LL + high-freq skip connections
                ▼
         Stylized Image
```

**Key components:**

| Class / Module | Description |
|---|---|
| `HaarPool2D` | Depthwise convolution with frozen Haar filters; splits input into 4 sub-band feature maps (2× spatial downsampling) |
| `HaarUnpool2D` | Transposed convolution reconstruction; sums all four sub-bands |
| `Encoder` | VGG-inspired: `3→64→128→256→512` channels, three `HaarPool2D` stages |
| `Decoder` | Mirrored decoder with three `HaarUnpool2D` stages |
| `NST` | Wraps encoder + decoder; runs per-level WCT feature transfer |
| `utils/core.py` | SVD-based WCT: whitens content features, recolors with style covariance; supports optional segmentation masks |
| `utils/io.py` | Image loading / saving, segmentation map parsing |

---

## Requirements

```
torch>=1.8.0
torchvision>=0.9.0
numpy
Pillow
tqdm
```

Install:

```bash
pip install torch torchvision numpy Pillow tqdm
```

---

## Usage

### Style Transfer (inference)

Pre-trained weights (`data/encoder.pth`, `data/decoder.pth`) are included in the repository.

```bash
python HaarPool.py
```

A file picker dialog will open twice — first to select the **content image**, then the **style image**.  
The result is saved as `stylized_image.png` in the working directory.

The blending strength is controlled by `alpha` (default `0.2`):
- `0.0` → original content, no style
- `1.0` → full style transfer

### Training

```bash
python training.py
```

- Downloads **CIFAR-10** automatically via torchvision
- Trains for **10 epochs**, batch size **64**
- Loss: `0.6 × MSE content loss + 0.4 × Gram-matrix style loss`
- Optimizer: Adam, `lr=0.001`
- Saves weights to `data/encoder.pth` and `data/decoder.pth`

Uses CUDA if available, otherwise falls back to CPU.

---

## Project Structure

```
WaveletStyleTransfer/
├── HaarPool.py          # Model definition + inference entry point
├── training.py          # Training script
├── data/
│   ├── encoder.pth      # Pre-trained encoder weights
│   └── decoder.pth      # Pre-trained decoder weights
└── utils/
    ├── __init__.py
    ├── core.py          # WCT algorithm (whitening & coloring transform)
    └── io.py            # Image I/O and segmentation map utilities
```

---

## Segmentation-guided Style Transfer

`utils/core.py` supports **semantic segmentation masks** to apply style per region.  
Pass segmentation PNG paths to `feature_wct()` — 9 color-coded labels are supported (sky, water, ground, vegetation, building, mountain, person, object, foreground).  
If no masks are provided, global WCT is applied to the entire image.

---

## Implementation Notes

- Haar wavelet filters are **mathematically fixed** — `requires_grad=False`. They are not learned.
- Inference runs on **CPU** by default (hardcoded in `NST`). Training uses CUDA if available.
- Input images are automatically resized and center-cropped to a multiple of 16 (required by three pooling stages).

---

## License

The core WCT implementation (`utils/core.py`, `utils/io.py`) is adapted from NVIDIA FastPhotoStyle and licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) — **non-commercial use only**.
