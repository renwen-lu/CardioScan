# CardioScan — ECG Digitization with ResNet-UNet Waveform Extraction
<img width="2752" height="1536" alt="31ffde8f9fdfe951fc515dc4242a7292" src="https://github.com/user-attachments/assets/d2c92d1d-21f2-492c-b530-1a068eafa620" />

Millions of electrocardiograms recorded over the past several decades exist only as paper printouts or scanned images, locked away in filing cabinets and hospital archives. They cannot be searched, compared, or fed into modern diagnostic algorithms. CardioScan converts a photograph or scan of any standard 12-lead ECG into calibrated digital time-series signals — making those records computationally accessible for the first time.

---

## The Problem

Paper ECGs represent an enormous untapped resource. A cardiologist reviewing a patient today has no practical way to compare their current rhythm against a tracing from five years ago if that earlier record exists only in print. Digitizing by hand is prohibitively slow. Existing commercial solutions are expensive, require proprietary hardware, and do not generalize well across the visual variety of ECG formats produced by different machines and institutions over the decades.

We built CardioScan to make digitization fast, accurate, and accessible — with nothing more than a photograph and a browser.

---

## How It Works

The pipeline runs three neural networks in sequence, each solving a distinct subproblem.

### Stage 0 — Orientation and Perspective Correction

The first network is a ResNet-18d encoder paired with a UNet decoder. It solves two tasks simultaneously: detecting the pixel locations of printed lead-name labels (I, II, III, aVR, aVL, aVF, V1–V6) via per-pixel semantic segmentation across 14 classes, and classifying the global rotation state of the image into one of eight orientations. The detected label positions serve as sparse keypoints. RANSAC homography maps these keypoints to a fixed reference coordinate system, producing a coarsely aligned image of 1152×1440 pixels. Test-time augmentation across four flip variants is averaged to improve keypoint localization robustness.

### Stage 1 — Grid Detection and Non-Linear Rectification

The second network is a ResNet-34 encoder with a UNet decoder trained on four simultaneous tasks: locating grid intersection points, assigning each pixel to one of 44 horizontal grid lines, assigning each pixel to one of 57 vertical grid lines, and re-detecting lead-name positions for cross-validation. The key insight is that labeling each pixel with a specific line index — rather than simply predicting binary foreground/background — allows the intersection coordinates to be read off directly by looking up which horizontal and vertical line each detected point belongs to. Missing intersections are recovered by cubic interpolation over the sparse 44×57 coordinate grid. The resulting dense flow field is used with `F.grid_sample` to apply a non-linear warp, correcting local paper curvature and lens distortion that a global homography cannot handle. Output is a physically calibrated image at 1700×2200 pixels.

### Stage 2 — Waveform Extraction

The third network is a ResNet-34 encoder paired with a coordinate-aware UNet decoder. Each decoder block concatenates normalized (x, y) coordinate maps to the feature tensor before convolution — a technique known as CoordConv — so the network can learn position-dependent behavior: the baseline voltage of row one is at a different pixel height than row two, and a translation-invariant convolution cannot distinguish them. The network outputs four probability maps, one per printed row, each predicting the vertical position of the waveform at every horizontal pixel. A dual-direction scan extracts the signal: for regions above the calibrated baseline, the scan proceeds top-down; for regions below, it proceeds bottom-up. This correctly handles both positive and negative deflections. The pixel coordinates are converted to millivolts using the physical calibration constants of the grid (78.8 pixels per millivolt). Savitzky-Golay filtering removes residual digitization noise. Einthoven's triangle correction (Lead II = Lead I + Lead III) redistributes the small inconsistencies introduced by independent lead extraction across the three limb leads.

---

## Source-Aware Preprocessing

ECG images in the real world vary dramatically in their visual characteristics depending on the recording machine, paper type, scanner settings, and photographic conditions. A single preprocessing pipeline that works well for a cleanly scanned hospital ECG will degrade results on a smartphone photograph taken under uneven lighting.

We trained an EfficientNet-B5 classifier on labeled examples from 12 distinct source categories. At inference time, each image is first classified by source. The predicted category selects a tailored preprocessing branch — combinations of CLAHE contrast enhancement, gray-world white balance, bilateral denoising, median filtering, and morphological background correction applied in LAB color space. Both the preprocessed image and the original are run through Stages 0 and 1, and the result with the higher quality score (a weighted combination of edge density and gradient anisotropy) is selected for Stage 2.

---

## Data

Training data was drawn from the PhysioNet ECG Image Digitization dataset, a public collection of 12-lead ECG images paired with ground-truth digital signals. The dataset spans recordings from multiple institutions and device manufacturers, covering the range of visual formats and print qualities encountered in clinical practice. Ground-truth signals are provided in CSV format at the original sampling rate of each recording.

The source classifier was trained on image-level labels derived from the filename convention embedded in the dataset identifiers. The waveform extraction model was supervised with pixel-level ground-truth maps generated by rendering the known signal positions onto the rectified image grid.

---

## Technical Stack

| Component | Technology |
|---|---|
| Stage 0 model | ResNet-18d + UNet, timm |
| Stage 1 model | ResNet-34 + UNet, timm |
| Stage 2 model | ResNet-34 + CoordConv UNet, timm |
| Source classifier | EfficientNet-B5, timm |
| Image processing | OpenCV, scipy, connected-components-3d |
| Serving backend | FastAPI, uvicorn, ngrok |
| Frontend | Vanilla HTML/CSS/JS, Canvas API |
| Training infrastructure | NVIDIA T4 GPU (Google Colab) |

---

## Challenges

**Waveform position ambiguity.** A standard UNet has no inherent sense of where in the image it is operating. When extracting four rows of ECG waveforms that span the full image height, the network needs to behave differently at y=300 than at y=1200. Injecting normalized coordinate channels at each decoder stage — CoordConv — gave the network the spatial context it needed and substantially reduced cross-row confusion.

**Peak and trough asymmetry.** ECG signals deflect both above and below the isoelectric baseline. A naive argmax of the probability map selects the single most confident pixel per column, which fails for broad, low-amplitude deflections where the true signal boundary is at the edge of the activation region rather than its center. The dual-direction scan — top-down for positive deflections, bottom-up for negative — correctly captures both cases and is robust to the mild class imbalance between peaks and troughs.

---

## Running the Demo

1. Open `backend/ecg_digitizer.ipynb` in Google Colab with a GPU runtime.
2. Follow the setup cells to configure the Kaggle API and download model weights.
3. Paste your ngrok authtoken and run the server cell. Copy the printed public URL.
4. Open `frontend/ecg_demo_v4.html` in any browser, paste the URL into the API field, and upload an ECG image.

Model weights are not included in this repository due to file size. Download instructions are provided in the notebook.

---

## Repository Structure

```
CardioScan/
├── model/
│   ├── stage0_model.py          ResNet-18d + UNet keypoint detector
│   ├── stage0_common.py         TTA, homography alignment, keypoint post-processing
│   ├── stage1_model.py          ResNet-34 + UNet grid detection network
│   ├── stage1_common.py         Grid reconstruction, dense warp, line cleaning
│   ├── stage2_model.py          ResNet-34 + CoordConv UNet waveform extractor
│   └── stage2_common.py         Pixel-to-series conversion, signal calibration
├── backend/
│   ├── ecg_digitizer.ipynb      FastAPI inference server (Colab + ngrok)
│   └── physionet-image-multi-class-train.ipynb   Source classifier training
└── frontend/
    └── ecg_demo_v4.html         CardioScan web interface
```
