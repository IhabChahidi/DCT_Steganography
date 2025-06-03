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
    block_variances = [[] for _ in range(3)]
    for bi in range(num_blocks_i):
        for bj in range(num_blocks_j):
            for c, channel in enumerate([b, g, r]):
                block = channel[bi*8:(bi+1)*8, bj*8:(bj+1)*8]
                dct_blocks[c].append(cv2.dct(block.astype(np.float32)))
                block_variances[c].append(compute_block_variance(block))
    logging.info(f"DCT computed for {len(dct_blocks[0])} blocks per channel")

    # Select mid-frequency coefficients
    selected_uv = [(u, v) for u in range(1, 8) for v in range(1, 8)]
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
    # AES IV is 16 bytes = 128 bits. encrypted_bytes = IV + actual_encrypted_data. So msg_bits includes IV bits.
    logging.info(f"AES IV is 128 bits. Encrypted message (with padding) is {L - 128} bits. Total L = {L} bits.")
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
        for k, idx in enumerate(idx_list):
            bi, bj, c, u, v = pool[idx]
            # Adaptive alpha based on block variance
            variance = block_variances[c][bi * num_blocks_j + bj]
            adaptive_alpha = alpha * (1 + variance / 1000)  # Scale alpha with variance
            # adaptive_alpha = min(adaptive_alpha, 1.0)  # Cap at 1.0 - REMOVED
            dct_blocks[c][bi * num_blocks_j + bj][u, v] += adaptive_alpha * m_i * p[k]
        logging.info(f"Bit {i} embedded with adaptive_alpha={adaptive_alpha:.4f}, K={len(idx_list)}")

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
    
    num_blocks_i, num_blocks_j = h // 8, w // 8

    # Calculate Block Variances
    block_variances_channels = [[] for _ in range(3)]
    for c_idx, channel_data in enumerate([b, g, r]):
        for bi_var in range(num_blocks_i):
            for bj_var in range(num_blocks_j):
                block_var = channel_data[bi_var*8:(bi_var+1)*8, bj_var*8:(bj_var+1)*8]
                variance = compute_block_variance(block_var)
                block_variances_channels[c_idx].append(variance)
    logging.info(f"Block variances computed for {len(block_variances_channels[0])} blocks per channel.")

    # Divide into 8x8 blocks and apply DCT for each channel
    dct_blocks = [[] for _ in range(3)]
    for bi_dct in range(num_blocks_i):
        for bj_dct in range(num_blocks_j):
            for c_dct, channel_data_dct in enumerate([b, g, r]): # Use different iter var names
                block_dct = channel_data_dct[bi_dct*8:(bi_dct+1)*8, bj_dct*8:(bj_dct+1)*8]
                dct_blocks[c_dct].append(cv2.dct(block_dct.astype(np.float32)))
    logging.info(f"DCT computed for {len(dct_blocks[0])} blocks per channel")

    # Define coefficient pool
    selected_uv = [(u, v) for u in range(1, 8) for v in range(1, 8)]
    pool = [(bi, bj, c, u, v) for bi in range(num_blocks_i)
            for bj in range(num_blocks_j) for c in range(3) for (u, v) in selected_uv]
    N = len(pool)
    logging.info(f"Coefficient pool size: {N}")

    # Extract length (28 encoded bits for 16-bit length)
    len_encoded_extracted = []
    current_K = K # K is the initial value from args
    max_attempts = 15 # Increased max_attempts for more alpha/K adjustments
    attempt = 0

    # Initialize Adaptive Parameters for length extraction
    current_alpha_estimate = 1.0
    max_alpha_estimate = 5.0
    min_alpha_estimate = 0.1
    alpha_adjustment_factor = 1.5
    change_alpha_direction_threshold = 2
    alpha_increases_done = 0
    alpha_decreases_done = 0
    alpha_adjust_direction = 1 # 1 for increase, -1 for decrease

    logging.info(f"Starting length extraction. Initial K={current_K}, Initial AlphaEst={current_alpha_estimate:.2f}")
    while attempt < max_attempts:
        len_encoded_extracted = []
        for i in tqdm(range(28), desc=f"Extracting length (attempt {attempt+1}/{max_attempts}, K={current_K}, alpha_est={current_alpha_estimate:.2f})"):
            random.seed(key + i)
            idx_list = random.sample(range(N), min(current_K, N))
            p = [random.choice([1, -1]) for _ in range(len(idx_list))]
            
            sum_weighted_signal_numerator = 0.0
            sum_weights_denominator = 0.0
            # num_coeffs_processed = 0 # No longer directly used for averaging, but can be kept for other K metrics if needed.
                                     # For now, removing to simplify, as K itself is logged.
            for k_loop_idx, pool_idx in enumerate(idx_list):
                bi_pool, bj_pool, c_channel_idx, u, v = pool[pool_idx]
                block_linear_idx = bi_pool * num_blocks_j + bj_pool
                variance = block_variances_channels[c_channel_idx][block_linear_idx]

                if variance < 1e-3: # Skip coefficients from very low-variance blocks
                    logging.debug(f"Bit {i}: Skipping coefficient from block (bi={bi_pool}, bj={bj_pool}, c={c_channel_idx}) due to very low variance: {variance:.4f}")
                    continue
                
                adaptive_component = (1 + variance / 1000.0)
                adaptive_component = max(adaptive_component, 0.01)

                dct_coeff_val = dct_blocks[c_channel_idx][block_linear_idx][u, v]
                pattern_val = p[k_loop_idx]
                
                sum_weighted_signal_numerator += dct_coeff_val * pattern_val
                sum_weights_denominator += adaptive_component
                # num_coeffs_processed += 1 # If re-added for other metrics
            
            avg_normalized_signal = sum_weighted_signal_numerator / sum_weights_denominator if sum_weights_denominator > 0 else 0.0
            bit = 1 if avg_normalized_signal > 0.0 else 0
            len_encoded_extracted.append(bit)
            logging.info(f"Length bit {i}, avg_norm_signal={avg_normalized_signal:.4f}, bit={bit}, K={current_K}, alpha_est={current_alpha_estimate:.2f}")

        # Decode length
        len_bits_extracted = []
        for j in range(0, 28, 7):
            chunk = len_encoded_extracted[j:j+7]
            len_bits_extracted.extend(hamming_decode(chunk))
        L = int(''.join(map(str, len_bits_extracted[:16])), 2)
        logging.info(f"Extracted length (decoded): {L}")

        # Validate length and ensure it's a multiple of 128 bits (16 bytes)
        if L > 0 and L <= 1024 and L % 128 == 0:  # Ensure L corresponds to a multiple of 16 bytes
            logging.info(f"Extracted valid length {L} on attempt {attempt+1}")
            break
        
        attempt += 1
        logging.info(f"Attempt {attempt} failed to extract valid length. Current K={current_K}, alpha_est={current_alpha_estimate:.2f}")
        
        K_ADJUSTMENT_ATTEMPTS = (max_attempts // 3) * 2
        if attempt < K_ADJUSTMENT_ATTEMPTS or attempt % 3 != 0:
            current_K = min(current_K * 2, N)
        else:
            if alpha_adjust_direction == 1:
                current_alpha_estimate = min(current_alpha_estimate * alpha_adjustment_factor, max_alpha_estimate)
                alpha_increases_done += 1
                if alpha_increases_done >= change_alpha_direction_threshold or current_alpha_estimate >= max_alpha_estimate:
                    alpha_adjust_direction = -1
                    alpha_decreases_done = 0 
            else: # alpha_adjust_direction == -1
                current_alpha_estimate = max(current_alpha_estimate / alpha_adjustment_factor, min_alpha_estimate)
                alpha_decreases_done += 1
                if alpha_decreases_done >= change_alpha_direction_threshold or current_alpha_estimate <= min_alpha_estimate:
                    alpha_adjust_direction = 1
                    alpha_increases_done = 0
        logging.info(f"Adjusting parameters for next attempt: new K={current_K}, new alpha_est={current_alpha_estimate:.2f}")


    if attempt >= max_attempts:
        logging.error(f"Failed to extract valid length after {max_attempts} attempts. Final K={current_K}, alpha_est={current_alpha_estimate:.2f}")
        raise ValueError(f"Failed to extract valid length after {max_attempts} attempts. Max K reached: {current_K}, final alpha estimate: {current_alpha_estimate:.2f}. Ensure key is correct and image is not overly distorted.")

    # Extract message bits
    num_chunks = (L + 3) // 4
    M_msg = num_chunks * 7
    msg_encoded_extracted = []
    attempt = 0 # Reset attempt counter for message extraction
    # Reset K to initial K for message extraction, alpha parameters are also reset
    current_K = K 

    # Initialize Adaptive Parameters for message extraction
    current_alpha_estimate = 1.0
    max_alpha_estimate = 5.0
    min_alpha_estimate = 0.1
    alpha_adjustment_factor = 1.5
    change_alpha_direction_threshold = 2
    alpha_increases_done = 0
    alpha_decreases_done = 0
    alpha_adjust_direction = 1

    logging.info(f"Starting message extraction for L={L}. Initial K={current_K}, Initial AlphaEst={current_alpha_estimate:.2f}")
    while attempt < max_attempts: # Use the same max_attempts for message part
        msg_encoded_extracted = []
        for i in tqdm(range(28, 28 + M_msg), desc=f"Extracting message (attempt {attempt+1}/{max_attempts}, K={current_K}, alpha_est={current_alpha_estimate:.2f})"):
            random.seed(key + i)
            idx_list = random.sample(range(N), min(current_K, N))
            p = [random.choice([1, -1]) for _ in range(len(idx_list))]

            sum_weighted_signal_numerator = 0.0
            sum_weights_denominator = 0.0
            # num_coeffs_processed = 0 # As above, removed for now
            for k_loop_idx, pool_idx in enumerate(idx_list):
                bi_pool, bj_pool, c_channel_idx, u, v = pool[pool_idx]
                block_linear_idx = bi_pool * num_blocks_j + bj_pool
                variance = block_variances_channels[c_channel_idx][block_linear_idx]

                if variance < 1e-3: # Skip coefficients from very low-variance blocks
                    logging.debug(f"Bit {i-28}: Skipping coefficient from block (bi={bi_pool}, bj={bj_pool}, c={c_channel_idx}) due to very low variance: {variance:.4f}")
                    continue

                adaptive_component = (1 + variance / 1000.0)
                adaptive_component = max(adaptive_component, 0.01)

                dct_coeff_val = dct_blocks[c_channel_idx][block_linear_idx][u, v]
                pattern_val = p[k_loop_idx]

                sum_weighted_signal_numerator += dct_coeff_val * pattern_val
                sum_weights_denominator += adaptive_component
                # num_coeffs_processed += 1 # If re-added
            
            avg_normalized_signal = sum_weighted_signal_numerator / sum_weights_denominator if sum_weights_denominator > 0 else 0.0
            bit = 1 if avg_normalized_signal > 0.0 else 0
            msg_encoded_extracted.append(bit)
            logging.info(f"Message bit {i-28}, avg_norm_signal={avg_normalized_signal:.4f}, bit={bit}, K={current_K}, alpha_est={current_alpha_estimate:.2f}")

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
                msg_extracted = message_bytes.decode('utf-8', errors='ignore') # Ignore errors for now, CRC is the main check
                # Basic check if message seems plausible, can be improved
                if len(msg_extracted) > 0 and all(32 <= ord(char) <= 126 or char == '\n' or char == '\r' for char in msg_extracted): # Allow more chars
                    logging.info(f"CRC verified successfully for message on attempt {attempt+1}")
                    break 
                else:
                    logging.warning(f"CRC matched but message content seems invalid on attempt {attempt+1}: {msg_extracted[:30]}... K={current_K}, AlphaEst={current_alpha_estimate:.2f}") # Log snippet
            else:
                logging.warning(f"Attempt {attempt+1}/{max_attempts}: CRC mismatch. Received={received_crc}, Computed={computed_crc}. K={current_K}, AlphaEst={current_alpha_estimate:.2f}")
        except (UnicodeDecodeError, ValueError, IndexError) as e: # Added IndexError for short messages
            logging.error(f"Attempt {attempt+1}/{max_attempts}: Message decoding error: {str(e)}. K={current_K}, AlphaEst={current_alpha_estimate:.2f}")

        attempt += 1
        logging.info(f"Attempt {attempt} failed to extract valid message. Current K={current_K}, alpha_est={current_alpha_estimate:.2f}")

        K_ADJUSTMENT_ATTEMPTS = (max_attempts // 3) * 2
        if attempt < K_ADJUSTMENT_ATTEMPTS or attempt % 3 != 0:
            current_K = min(current_K * 2, N)
        else:
            if alpha_adjust_direction == 1:
                current_alpha_estimate = min(current_alpha_estimate * alpha_adjustment_factor, max_alpha_estimate)
                alpha_increases_done += 1
                if alpha_increases_done >= change_alpha_direction_threshold or current_alpha_estimate >= max_alpha_estimate:
                    alpha_adjust_direction = -1
                    alpha_decreases_done = 0
            else: # alpha_adjust_direction == -1
                current_alpha_estimate = max(current_alpha_estimate / alpha_adjustment_factor, min_alpha_estimate)
                alpha_decreases_done += 1
                if alpha_decreases_done >= change_alpha_direction_threshold or current_alpha_estimate <= min_alpha_estimate:
                    alpha_adjust_direction = 1
                    alpha_increases_done = 0
        logging.info(f"Adjusting parameters for next attempt: new K={current_K}, new alpha_est={current_alpha_estimate:.2f}")


    if attempt >= max_attempts:
        logging.error(f"Failed to extract valid message after {max_attempts} attempts (L={L}). Final K={current_K}, alpha_est={current_alpha_estimate:.2f}")
        raise ValueError(f"Failed to extract valid message after {max_attempts} attempts (L={L}). Max K reached: {current_K}, final alpha estimate: {current_alpha_estimate:.2f}. CRC or decoding failed. Ensure key is correct and image integrity.")

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
    embed_parser.add_argument("--alpha", type=float, default=1.0, help="Initial embedding strength (default: 1.0)")
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