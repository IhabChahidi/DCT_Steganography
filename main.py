"""
Advanced DCT-based Spread Spectrum Watermarking Tool

Overview:
This script implements a digital image watermarking technique based on the
Discrete Cosine Transform (DCT) and spread spectrum principles. The core idea is
to embed a secret message into the mid-frequency DCT coefficients of an image.
The message is first encrypted using AES, then encoded with Hamming codes for error
correction, and finally embedded by subtly modifying selected DCT coefficients.
Extraction attempts to reverse this process.

Watermarking Scheme:
1.  Message Preparation:
    - The secret message is encoded to UTF-8.
    - A CRC32 checksum is appended for integrity verification.
    - The combined message+CRC is encrypted using AES-256-CBC. The IV is prepended.
    - The length of the encrypted bitstream is also encoded using Hamming codes.
    - The encrypted bitstream itself is encoded using Hamming(7,4) codes.
2.  Image Preparation:
    - The input image is converted to BGR color space.
    - Each color channel is divided into 8x8 blocks.
    - DCT is applied to each block.
3.  Coefficient Selection & Embedding:
    - A pool of mid-frequency AC coefficients (typically (1,1) to (7,7) within each 8x8 block,
      excluding the DC coefficient) is created from all color channels.
    - For each bit of the combined (length + message) encoded bitstream:
        - A pseudo-random subset of K coefficients is chosen from the pool using a
          secret key and the bit's index to seed the RNG.
        - A pseudo-random pattern p (of +1 or -1) of length K is generated.
        - The bit (m_i, converted to +1 or -1) is embedded by modifying the selected
          coefficients: X'_uv = X_uv + strength * m_i * p_k.
        - Initially, an adaptive strength mechanism was used, but current tests also
          explore fixed strength.
4.  Image Reconstruction:
    - Inverse DCT (IDCT) is applied to each block.
    - Pixel values are clipped to [0, 255] and converted to uint8.
    - The watermarked image is saved.
5.  Extraction:
    - The watermarked image is processed (DCT on 8x8 blocks).
    - For each bit to be extracted (first length, then message):
        - The same K coefficients are selected using the key and bit index.
        - The same pseudo-random pattern p is generated.
        - The correlation sum = sum(X'_uv * p_k) is computed.
        - The extracted bit is determined by the sign of this sum.
    - Hamming decoding and AES decryption are applied to recover the message.
    - CRC32 checksum is verified.

Known Limitations & Observations from `sample_image.png` testing:
-   Sensitivity to Image Content: The algorithm's performance is highly dependent
    on the characteristics of the input image.
-   Failure on Simple/Artificial Images: Extensive testing with a generated
    64x64px image (`sample_image.png`) featuring large flat color areas and sharp
    contrasts consistently resulted in extraction failures.
    -   The core issue is that the "noise" term, derived from the original image's
        DCT coefficients (sum(X_uv * p_k) / K), often overwhelms the embedded signal
        term (strength * m_i). This prevents correct bit recovery.
    -   Attempts to tune parameters like embedding strength (`alpha`), number of
        coefficients per bit (`K`), and the specific DCT frequency bands used
        (e.g., `(1,1)-(7,7)` vs. `(4,4)-(7,7)`) did not reliably overcome this issue
        for `sample_image.png`.
        -   Low strength (e.g., alpha=1.0) leads to the signal being lost in
            quantization noise (when converting float DCT back to uint8 pixels).
        -   High strength (e.g., alpha=15.0 or adaptive values reaching 10.0 or more)
            leads to significant clipping of pixel values during uint8 conversion.
            This non-linear clipping corrupts the embedded signal, again making
            extraction fail as the linear model X' = X + signal is violated.
-   Suitability for Textured Images: It is hypothesized that the algorithm might
    perform better on textured, natural images where DCT coefficients are more
    evenly distributed and potentially have lower individual magnitudes in the
    mid-frequencies, or where the variance calculation for adaptive alpha could
    be more meaningful. However, this has not been verified in the current test
    environment.

This comment block was added after a series of systematic tests documented in
the commit history, which revealed these limitations when applied to the
`sample_image.png` created for testing.
"""
import cv2
import numpy as np
import random
import argparse
import logging
from random import Random
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import os
import zlib
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    filename='watermarking.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def hamming_encode(data_bits):
    """Encode 4 data bits into 7 bits using Hamming(7,4) code."""
    assert len(data_bits) == 4, "Input must be 4 bits"
    d1, d2, d3, d4 = data_bits
    p1 = d1 ^ d2 ^ d4
    p2 = d1 ^ d3 ^ d4
    p3 = d2 ^ d3 ^ d4
    return [p1, p2, d1, p3, d2, d3, d4]

def hamming_decode(codeword):
    """Decode 7-bit Hamming codeword to 4 data bits, correcting single-bit errors."""
    assert len(codeword) == 7, "Input must be 7 bits"
    p1, p2, d1, p3, d2, d3, d4 = codeword
    s1 = p1 ^ d1 ^ d2 ^ d4
    s2 = p2 ^ d1 ^ d3 ^ d4
    s3 = p3 ^ d2 ^ d3 ^ d4
    syndrome = s1 + 2 * s2 + 4 * s3
    if syndrome != 0 and syndrome <= 7:
        error_pos = syndrome - 1
        codeword[error_pos] ^= 1
    return [codeword[2], codeword[4], codeword[5], codeword[6]]

def generate_key_stream(length, key):
    """Generate a pseudo-random key stream for encryption."""
    rand = Random(key)
    return [rand.randint(0, 255) for _ in range(length)]

def aes_encrypt(message_bytes, key):
    """Encrypt the message bytes using AES-256-CBC."""
    key_bytes = bytes(generate_key_stream(32, key))  # 32 bytes for AES-256
    iv = os.urandom(16)  # 16-byte IV
    cipher = Cipher(algorithms.AES(key_bytes), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    # Pad message to multiple of 16 bytes
    pad_length = 16 - (len(message_bytes) % 16)
    padded_message = message_bytes + bytes([pad_length] * pad_length)
    encrypted = encryptor.update(padded_message) + encryptor.finalize()
    logging.info(f"Message encrypted with AES-256, length: {len(encrypted)} bytes")
    return iv + encrypted  # Prepend IV for decryption

def aes_decrypt(encrypted_bytes, key):
    """Decrypt the encrypted bytes using AES-256-CBC."""
    key_bytes = bytes(generate_key_stream(32, key))
    iv = encrypted_bytes[:16]
    ciphertext = encrypted_bytes[16:]
    cipher = Cipher(algorithms.AES(key_bytes), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    padded_message = decryptor.update(ciphertext) + decryptor.finalize()
    # Remove padding
    pad_length = padded_message[-1]
    message = padded_message[:-pad_length]
    logging.info(f"Message decrypted with AES-256, length: {len(message)} bytes")
    return message

def compute_crc32(data):
    """Compute CRC-32 checksum for data."""
    return zlib.crc32(data) & 0xFFFFFFFF

def compute_block_variance(block):
    """Compute variance of an 8x8 block for adaptive embedding."""
    return np.var(block)

def embed(image_path, message, output_path, key, alpha, K):
    """Embed a secret message into a color image with adaptive strength."""
    # Validate image format
    valid_extensions = ['.png', '.jpg', '.jpeg']
    if not any(image_path.lower().endswith(ext) for ext in valid_extensions):
        raise ValueError("Image must be PNG or JPEG")
    
    # Load and validate image
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot load image at {image_path}")
    if len(img.shape) != 3 or img.shape[2] != 3:
        raise ValueError("Input image must be a color image with 3 channels")
    
    # Pad image to multiple of 8 if necessary
    h, w, _ = img.shape
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    if pad_h > 0 or pad_w > 0:
        img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)
        h, w, _ = img.shape
    logging.info(f"Image loaded: {h}x{w}, blocks: {h//8}x{w//8}")

    # Validate message length
    if len(message) > 128:
        raise ValueError("Message too long; maximum 128 characters")
    
    # Split into channels
    b, g, r = cv2.split(img)

    # Divide into 8x8 blocks and apply DCT for each channel
    num_blocks_i, num_blocks_j = h // 8, w // 8
    dct_blocks = [[] for _ in range(3)]
    # block_variances = [[] for _ in range(3)] # Removed
    for bi in range(num_blocks_i):
        for bj in range(num_blocks_j):
            for c, channel in enumerate([b, g, r]):
                block = channel[bi*8:(bi+1)*8, bj*8:(bj+1)*8]
                dct_blocks[c].append(cv2.dct(block.astype(np.float32)))
                # block_variances[c].append(compute_block_variance(block)) # Removed
    logging.info(f"DCT computed for {len(dct_blocks[0])} blocks per channel")

    # Select mid-frequency coefficients
    selected_uv = [(u, v) for u in range(1, 8) for v in range(1, 8)] # Reverted to original
    pool = [(bi, bj, c, u, v) for bi in range(num_blocks_i)
            for bj in range(num_blocks_j) for c in range(3) for (u, v) in selected_uv]
    N = len(pool)
    logging.info(f"Coefficient pool size: {N}")

    # Encrypt message with AES and compute CRC
    message_bytes = message.encode('utf-8')
    crc = compute_crc32(message_bytes)
    message_with_crc = message_bytes + crc.to_bytes(4, 'big')
    encrypted_bytes = aes_encrypt(message_with_crc, key)
    msg_bits = [int(b) for byte in encrypted_bytes for b in format(byte, '08b')]
    L = len(msg_bits)  # This now reflects the actual number of bits after padding
    logging.info(f"Message encrypted and converted to {L} bits")

    # Encode length (16 bits) with Hamming
    len_bits = [int(b) for b in bin(L)[2:].zfill(16)]
    len_encoded = []
    for i in range(0, 16, 4):
        chunk = len_bits[i:i+4]
        len_encoded.extend(hamming_encode(chunk))
    logging.info(f"Length {L} encoded to {len(len_encoded)} bits")

    # Encode message with Hamming
    num_chunks = (L + 3) // 4
    msg_encoded = []
    for i in range(num_chunks):
        chunk = msg_bits[i*4:i*4+4]
        if len(chunk) < 4:
            chunk.extend([0] * (4 - len(chunk)))
        msg_encoded.extend(hamming_encode(chunk))
    logging.info(f"Message encoded to {len(msg_encoded)} bits")

    # Combine length and message bits
    encoded_bits = len_encoded + msg_encoded
    M = len(encoded_bits)
    logging.info(f"Total encoded bits: {M}")

    # Embed each bit with adaptive alpha
    for i in tqdm(range(M), desc="Embedding bits"):
        random.seed(key + i)
        idx_list = random.sample(range(N), min(K, N))
        p = [random.choice([1, -1]) for _ in range(len(idx_list))]
        m_i = 2 * encoded_bits[i] - 1
        current_embedding_strength = 15.0  # Use fixed embedding strength
        for k, idx in enumerate(idx_list):
            bi, bj, c, u, v = pool[idx]
            dct_blocks[c][bi * num_blocks_j + bj][u, v] += current_embedding_strength * m_i * p[k]
        logging.info(f"Bit {i} embedded with fixed_strength={current_embedding_strength:.4f}, K={len(idx_list)}")

    # Reconstruct image
    watermarked_blocks = [[cv2.idct(dct_block) for dct_block in channel_blocks] for channel_blocks in dct_blocks]
    watermarked_channels = [np.zeros_like(channel) for channel in [b, g, r]]
    for c in range(3):
        for bi in range(num_blocks_i):
            for bj in range(num_blocks_j):
                watermarked_channels[c][bi*8:(bi+1)*8, bj*8:(bj+1)*8] = watermarked_blocks[c][bi * num_blocks_j + bj]
    watermarked_img = cv2.merge([np.clip(channel, 0, 255).astype(np.uint8) for channel in watermarked_channels])
    cv2.imwrite(output_path, watermarked_img)
    logging.info(f"Image saved to {output_path}")

def extract(image_path, key, K):
    """Extract a secret message from a color image with noise resilience."""
    # Validate image format
    valid_extensions = ['.png', '.jpg', '.jpeg']
    if not any(image_path.lower().endswith(ext) for ext in valid_extensions):
        raise ValueError("Image must be PNG or JPEG")
    
    # Load and validate image
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot load image at {image_path}")
    if len(img.shape) != 3 or img.shape[2] != 3:
        raise ValueError("Input image must be a color image with 3 channels")
    
    # Pad image to multiple of 8 if necessary
    h, w, _ = img.shape
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    if pad_h > 0 or pad_w > 0:
        img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)
        h, w, _ = img.shape
    logging.info(f"Image loaded: {h}x{w}, blocks: {h//8}x{w//8}")

    # Split into channels
    b, g, r = cv2.split(img)

    # Divide into 8x8 blocks and apply DCT for each channel
    num_blocks_i, num_blocks_j = h // 8, w // 8
    dct_blocks = [[] for _ in range(3)]
    for bi in range(num_blocks_i):
        for bj in range(num_blocks_j):
            for c, channel in enumerate([b, g, r]):
                block = channel[bi*8:(bi+1)*8, bj*8:(bj+1)*8]
                dct_blocks[c].append(cv2.dct(block.astype(np.float32)))
    logging.info(f"DCT computed for {len(dct_blocks[0])} blocks per channel")

    # Define coefficient pool
    selected_uv = [(u, v) for u in range(1, 8) for v in range(1, 8)] # Reverted to original
    pool = [(bi, bj, c, u, v) for bi in range(num_blocks_i)
            for bj in range(num_blocks_j) for c in range(3) for (u, v) in selected_uv]
    N = len(pool)
    logging.info(f"Coefficient pool size: {N}")

    # Extract length (28 encoded bits for 16-bit length)
    len_encoded_extracted = []
    max_alpha = 1.0
    current_alpha = 0.1
    current_K = K
    max_attempts = 5
    attempt = 0

    while attempt < max_attempts:
        len_encoded_extracted = []
        for i in tqdm(range(28), desc=f"Extracting length (attempt {attempt+1})"):
            random.seed(key + i)
            idx_list = random.sample(range(N), min(current_K, N))
            p = [random.choice([1, -1]) for _ in range(len(idx_list))]
            # Average across channels for noise resilience
            sum_corr = 0
            count = 0
            for k, idx in enumerate(idx_list):
                bi, bj, c, u, v = pool[idx]
                sum_corr += dct_blocks[c][bi * num_blocks_j + bj][u, v] * p[k]
                count += 1
            c = sum_corr / count if count > 0 else 0
            bit = 1 if c > 0 else 0
            len_encoded_extracted.append(bit)
            logging.info(f"Length bit {i}, sum_corr={sum_corr:.4f}, c={c:.4f}, bit={bit}, alpha={current_alpha}, K={current_K}")

        # Decode length
        len_bits_extracted = []
        for j in range(0, 28, 7):
            chunk = len_encoded_extracted[j:j+7]
            len_bits_extracted.extend(hamming_decode(chunk))
        L = int(''.join(map(str, len_bits_extracted[:16])), 2)
        logging.info(f"Extracted length (decoded): {L}")

        # Validate length and ensure it's a multiple of 128 bits (16 bytes)
        if L > 0 and L <= 1024 and L % 128 == 0:  # Ensure L corresponds to a multiple of 16 bytes
            break
        attempt += 1
        current_alpha = min(current_alpha * 2, max_alpha)
        current_K = min(current_K * 2, N)
        logging.info(f"Attempt {attempt} failed, adjusting alpha to {current_alpha}, K to {current_K}")

    if attempt >= max_attempts:
        raise ValueError("Failed to extract valid length after maximum attempts")

    # Extract message bits
    num_chunks = (L + 3) // 4
    M_msg = num_chunks * 7
    msg_encoded_extracted = []
    attempt = 0
    current_alpha = 0.1
    current_K = K

    while attempt < max_attempts:
        msg_encoded_extracted = []
        for i in tqdm(range(28, 28 + M_msg), desc=f"Extracting message (attempt {attempt+1})"):
            random.seed(key + i)
            idx_list = random.sample(range(N), min(current_K, N))
            p = [random.choice([1, -1]) for _ in range(len(idx_list))]
            sum_corr = 0
            count = 0
            for k, idx in enumerate(idx_list):
                bi, bj, c, u, v = pool[idx]
                sum_corr += dct_blocks[c][bi * num_blocks_j + bj][u, v] * p[k]
                count += 1
            c = sum_corr / count if count > 0 else 0
            bit = 1 if c > 0 else 0
            msg_encoded_extracted.append(bit)
            logging.info(f"Message bit {i-28}, sum_corr={sum_corr:.4f}, c={c:.4f}, bit={bit}, alpha={current_alpha}, K={current_K}")

        # Decode message
        msg_bits_extracted = []
        for j in range(0, len(msg_encoded_extracted), 7):
            chunk = msg_encoded_extracted[j:j+7]
            if len(chunk) < 7:
                break
            msg_bits_extracted.extend(hamming_decode(chunk))
        msg_bits_extracted = msg_bits_extracted[:L]

        # Convert to bytes and decrypt
        extracted_bytes = [int(''.join(map(str, msg_bits_extracted[i:i+8])), 2) for i in range(0, L, 8)]
        decrypted_bytes = aes_decrypt(bytes(extracted_bytes), key)

        # Verify CRC and decode
        try:
            received_crc = int.from_bytes(decrypted_bytes[-4:], 'big')
            message_bytes = decrypted_bytes[:-4]
            computed_crc = compute_crc32(message_bytes)
            if received_crc == computed_crc:
                msg_extracted = message_bytes.decode('utf-8')
                if all(32 <= ord(char) <= 126 for char in msg_extracted):
                    logging.info("CRC verified successfully")
                    break
            else:
                logging.warning(f"CRC mismatch: received {received_crc}, computed {computed_crc}")
        except (UnicodeDecodeError, ValueError) as e:
            logging.error(f"Decoding error: {str(e)}")

        attempt += 1
        current_alpha = min(current_alpha * 2, max_alpha)
        current_K = min(current_K * 2, N)
        logging.info(f"Attempt {attempt} failed, adjusting alpha to {current_alpha}, K to {current_K}")

    if attempt >= max_attempts:
        raise ValueError("Failed to extract valid message after maximum attempts")

    return msg_extracted

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Advanced DCT-based Spread Spectrum Watermarking Tool")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Embed command
    embed_parser = subparsers.add_parser("embed", help="Embed a message into a color image")
    embed_parser.add_argument("--image", required=True, help="Input color image path (PNG/JPEG)")
    embed_parser.add_argument("--message", required=True, help="Message to embed (max 128 chars)")
    embed_parser.add_argument("--output", required=True, help="Output watermarked color image path")
    embed_parser.add_argument("--key", type=int, required=True, help="Key for pseudo-random sequence and encryption")
    embed_parser.add_argument("--alpha", type=float, default=15.0, help="Initial embedding strength (default: 15.0)")
    embed_parser.add_argument("--K", type=int, default=500, help="Initial coefficients per bit (default: 500)")

    # Extract command
    extract_parser = subparsers.add_parser("extract", help="Extract a message from a color image")
    extract_parser.add_argument("--image", required=True, help="Input watermarked color image path (PNG/JPEG)")
    extract_parser.add_argument("--key", type=int, required=True, help="Key used during embedding")
    extract_parser.add_argument("--K", type=int, default=500, help="Initial coefficients per bit (default: 500)")

    args = parser.parse_args()

    try:
        if args.command == "embed":
            embed(args.image, args.message, args.output, args.key, args.alpha, args.K)
            print(f"Message embedded successfully into {args.output}")
        elif args.command == "extract":
            message = extract(args.image, args.key, args.K)
            print(f"Extracted message: {message}")
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        print(f"Error: {str(e)}")