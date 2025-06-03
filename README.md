# DCT_Steganography

This tool implements a DCT-based spread spectrum watermarking technique to embed and extract messages from color images.

## Features
* Embeds messages into color images (PNG/JPEG).
* Extracts messages from watermarked color images.
* Uses AES-256 encryption for the message.
* Employs Hamming codes for error correction of message length and content.
* Adaptive embedding strength based on block variance.
* Optional preprocessing for extraction to handle image artifacts.

## Setup
Ensure you have Python installed along with the following libraries:
* OpenCV (`cv2`)
* NumPy
* Cryptography
* tqdm

You can install them using pip:
```bash
pip install opencv-python numpy cryptography tqdm
```

## Usage
The tool has two main commands: `embed` and `extract`.

### Embed Command
This command embeds a secret message into an image.

**Arguments:**
*   `--image <path>`: (Required) Input color image path (PNG/JPEG).
*   `--message <text>`: (Required) Message to embed (max 128 printable ASCII characters).
*   `--output <path>`: (Required) Output watermarked color image path.
*   `--key <integer>`: (Required) Key for pseudo-random sequence generation and message encryption.
*   `--alpha <float>`: (Default: 1.0) Initial embedding strength.
*   `--K <integer>`: (Default: 500) Number of DCT coefficients used per bit.
*   `--preprocess`: (Placeholder for embed) Currently, no preprocessing is applied during embedding.

**Example:**
```bash
python main.py embed --image input.png --message "secret message" --output watermarked.png --key 12345
```

### Extract Command
This command extracts a secret message from a watermarked image.

**Arguments:**
*   `--image <path>`: (Required) Input watermarked color image path (PNG/JPEG).
*   `--key <integer>`: (Required) Key used during the embedding process.
*   `--K <integer>`: (Default: 500) Initial number of DCT coefficients to check per bit.
*   `--preprocess`: Enables image preprocessing steps before attempting to extract the watermark. This can be particularly useful for images that may have undergone compression (like JPEGs) or have minor artifacts.
*   `--blur-sigma <float>`: (Default: 0.7) Sets the `sigma` for the Gaussian blur applied during preprocessing. Active only if `--preprocess` is used and variance threshold is met.
*   `--clahe-clip-limit <float>`: (Default: 1.5) Sets the `clipLimit` for CLAHE (Contrast Limited Adaptive Histogram Equalization) during preprocessing. Active only if `--preprocess` is used and variance threshold is met.
*   `--variance-threshold <float>`: (Default: 1000.0) Preprocessing (blur and CLAHE) is only applied if the image's grayscale variance exceeds this value. Active only if `--preprocess` is used.

**Example:**
```bash
python main.py extract --image watermarked.png --key 12345 --preprocess --blur-sigma 0.8
```

### Image Preprocessing Details (for `extract` command)

The `--preprocess` flag for the `extract` command enables a sequence of image processing operations designed to improve the chances of successful watermark extraction, especially from images that might have been subjected to compression (e.g., JPEG) or other alterations that introduce artifacts.

When `--preprocess` is enabled, the following operations are conditionally applied:
1.  **Variance Check**: The system first calculates the variance of the grayscale version of the input image.
2.  **Gaussian Blur**: If the variance exceeds the `--variance-threshold`, a Gaussian blur is applied. This step aims to reduce high-frequency noise that might interfere with the watermark signal. The intensity of the blur is controlled by `--blur-sigma`.
3.  **CLAHE (Contrast Limited Adaptive Histogram Equalization)**: Following the blur, CLAHE is applied to the L-channel (lightness) of the image in LAB color space. This enhances local contrast, potentially making the embedded watermark signal more discernible. The `--clahe-clip-limit` parameter controls the extent of contrast enhancement.

**Parameters for Preprocessing:**

*   `--preprocess`: Enables the conditional preprocessing steps.
*   `--variance-threshold <value>`: (Default: 1000.0) This is the crucial threshold. Preprocessing (both blur and CLAHE) will *only* be performed if the calculated variance of the image's grayscale representation is greater than this specified value.
    *   *Rationale*: Images with very low variance (e.g., flat color areas) might not benefit from these operations or could even be negatively impacted. High variance often indicates more texture and detail, where noise reduction and contrast enhancement can be beneficial.
    *   *Adjustment*: Lower this value if you want to force preprocessing on images that have less natural variance but you still suspect could benefit from it.
*   `--blur-sigma <value>`: (Default: 0.7) Sets the `sigma` (standard deviation) for the Gaussian blur kernel.
    *   *Effect*: Higher values result in a more intense blur. This might be helpful for images with more pronounced noise.
    *   *Caution*: Excessive blurring can also attenuate the watermark signal itself, making extraction harder.
*   `--clahe-clip-limit <value>`: (Default: 1.5) Sets the contrast `clipLimit` for the CLAHE algorithm.
    *   *Effect*: Higher values lead to stronger local contrast enhancement.
    *   *Caution*: Overly aggressive contrast enhancement can amplify existing noise and artifacts, potentially obscuring the watermark.

**When to Adjust Preprocessing Parameters:**

While the default preprocessing settings are designed to be a reasonable starting point, you might need to experiment with these parameters if:
*   You are working with JPEG images that exhibit noticeable compression artifacts.
*   Default extraction (with or without `--preprocess`) fails, and you suspect image quality issues are the culprit.
*   You are processing a batch of images that share specific visual characteristics (e.g., all are low-light, all are heavily textured).
*   A lower `--variance-threshold` might be useful if your images are generally "flat" but you still want to try applying blur/CLAHE.
*   A slightly higher `--blur-sigma` could be attempted for images with visible noise, but increase cautiously.
*   Adjusting `--clahe-clip-limit` (e.g., between 1.0 and 3.0) can sometimes help if the default isn't effective.

**Limitations of Preprocessing:**

It's important to understand that preprocessing is an aid, not a guaranteed fix:
*   It is primarily designed to help with mild to moderate image artifacts or quality degradation.
*   For images that are very heavily compressed, severely corrupted, or have undergone complex transformations post-watermarking, preprocessing may not be sufficient to recover the message.
*   The ultimate success of watermark extraction always depends on factors like the strength of the original embedding, the correctness of the extraction key, and the overall integrity of the watermarked image data.

## Logging
The tool generates a `watermarking.log` file that records detailed information about the embedding and extraction processes, including parameter values, intermediate steps, and any errors encountered. This log is invaluable for debugging and understanding the behavior of the tool.
