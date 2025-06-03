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

def embed(image_path, message, output_path, key, alpha, K, preprocess=False):
    """Embed a secret message into a color image with adaptive strength."""
    logging.info(f"Embed called with --preprocess flag: {preprocess}") # Log preprocess flag state
    # Validate message for ASCII characters first
    for char_val in message:
        if not (32 <= ord(char_val) <= 126):
            raise ValueError("Error: Message must contain only printable ASCII characters (ordinal values 32-126).")
    logging.info("Input message validated: All characters are printable ASCII.")

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

# Helper function for message extraction attempts
def _extract_message_for_length(
    target_L,
    initial_K_for_msg,
    max_attempts_for_msg,
    key_param, # Renamed to avoid conflict with 'key' from outer scope if nested
    N_pool_size,
    pool_details,
    num_blocks_j_img,
    averaged_block_variances_data,
    averaged_dct_blocks_data,
    main_max_attempts_for_adaptive_params # For K_ADJUSTMENT_ATTEMPTS_MSG logic
):
    max_avg_bit_confidence_for_this_L_call = 0.0
    num_non_ascii_replacements_final = 0 # For this specific successful attempt

    msg_encoded_extracted = []
    current_K_msg = initial_K_for_msg

    current_alpha_estimate_msg = 1.0
    max_alpha_estimate_msg = 5.0
    min_alpha_estimate_msg = 0.1
    alpha_adjustment_factor_msg = 1.5
    change_alpha_direction_threshold_msg = 2
    alpha_increases_done_msg = 0
    alpha_decreases_done_msg = 0
    alpha_adjust_direction_msg = 1

    logging.info(f"Message Extraction Helper: Attempting L={target_L}, initial_K={current_K_msg}, max_attempts={max_attempts_for_msg}")

    for attempt_msg_num in range(max_attempts_for_msg):
        msg_encoded_extracted = []
        bit_confidences_this_attempt = [] # Store individual bit confidences for this attempt
        num_chunks_msg = (target_L + 3) // 4 # Ensure integer division
        M_bits_to_extract_msg = num_chunks_msg * 7

        desc_msg = f"Extracting L={target_L} (attempt {attempt_msg_num+1}/{max_attempts_for_msg}, K={current_K_msg}, alpha={current_alpha_estimate_msg:.2f})"
        for i_msg_bit_loop in tqdm(range(28, 28 + M_bits_to_extract_msg), desc=desc_msg, leave=False):
            random.seed(key_param + i_msg_bit_loop)
            idx_list_msg = random.sample(range(N_pool_size), min(current_K_msg, N_pool_size))
            p_msg = [random.choice([1, -1]) for _ in range(len(idx_list_msg))]

            sum_weighted_signal_numerator_msg = 0.0
            sum_weights_denominator_msg = 0.0
            for k_loop_idx_msg, pool_idx_msg in enumerate(idx_list_msg):
                bi_pool, bj_pool, u, v = pool_details[pool_idx_msg]
                block_linear_idx_msg = bi_pool * num_blocks_j_img + bj_pool
                variance_msg = averaged_block_variances_data[block_linear_idx_msg]

                if variance_msg < 1e-3:
                    logging.debug(f"Bit {i_msg_bit_loop-28} (L={target_L}): Skipping coeff from block (bi={bi_pool}, bj={bj_pool}) low var: {variance_msg:.4f}")
                    continue

                adaptive_component_msg = max((1 + variance_msg / 1000.0), 0.01)
                dct_coeff_val_msg = averaged_dct_blocks_data[block_linear_idx_msg][u, v]
                pattern_val_msg = p_msg[k_loop_idx_msg]
                sum_weighted_signal_numerator_msg += dct_coeff_val_msg * pattern_val_msg
                sum_weights_denominator_msg += adaptive_component_msg

            avg_normalized_signal_msg = sum_weighted_signal_numerator_msg / sum_weights_denominator_msg if sum_weights_denominator_msg > 0 else 0.0
            bit_msg = 1 if avg_normalized_signal_msg > 0.0 else 0
            msg_encoded_extracted.append(bit_msg)

            current_bit_confidence = min(abs(avg_normalized_signal_msg), 1.5) / 1.5
            bit_confidences_this_attempt.append(current_bit_confidence)

        avg_confidence_this_attempt = sum(bit_confidences_this_attempt) / len(bit_confidences_this_attempt) if bit_confidences_this_attempt else 0.0
        max_avg_bit_confidence_for_this_L_call = max(max_avg_bit_confidence_for_this_L_call, avg_confidence_this_attempt)
        logging.debug(f"Attempt {attempt_msg_num+1} for L={target_L}: Avg bit confidence {avg_confidence_this_attempt*100:.1f}%")

        msg_bits_extracted = []
        if len(msg_encoded_extracted) != M_bits_to_extract_msg:
             logging.warning(f"Attempt {attempt_msg_num+1} for L={target_L}: Encoded bits length {len(msg_encoded_extracted)} != expected {M_bits_to_extract_msg}. Skipping this attempt.")
        else:
            for j_msg in range(0, len(msg_encoded_extracted), 7):
                chunk_msg = msg_encoded_extracted[j_msg:j_msg+7]
                if len(chunk_msg) < 7: break
                msg_bits_extracted.extend(hamming_decode(chunk_msg))
            msg_bits_extracted = msg_bits_extracted[:target_L]

        if len(msg_bits_extracted) != target_L:
            logging.warning(f"Attempt {attempt_msg_num+1} for L={target_L}: Decoded bits length {len(msg_bits_extracted)} != target {target_L}.")
            # Fall through to K/Alpha adjustment, update max_avg_bit_confidence before potential K/Alpha adjustment
            max_avg_bit_confidence_for_this_L_call = max(max_avg_bit_confidence_for_this_L_call, avg_confidence_this_attempt)
        else:
            extracted_bytes_msg = [int(''.join(map(str, msg_bits_extracted[k_byte:k_byte+8])), 2) for k_byte in range(0, target_L, 8)]
            try:
                decrypted_bytes_msg = aes_decrypt(bytes(extracted_bytes_msg), key_param)
                received_crc_msg = int.from_bytes(decrypted_bytes_msg[-4:], 'big')
                message_bytes_for_content_check = decrypted_bytes_msg[:-4] # For content check
                computed_crc_msg = compute_crc32(message_bytes_for_content_check)

                if received_crc_msg == computed_crc_msg:
                    processed_message_bytes = bytearray()
                    non_ascii_replacements = 0
                    replaced_indices = []
                    final_msg_str = ""

                    if not message_bytes_for_content_check:
                        final_msg_str = "" # Empty message is valid
                    else:
                        for idx, byte_val in enumerate(message_bytes_for_content_check):
                            if not (32 <= byte_val <= 126): # Check for printable ASCII
                                processed_message_bytes.append(ord('?'))
                                non_ascii_replacements += 1
                                replaced_indices.append(idx)
                            else:
                                processed_message_bytes.append(byte_val)
                        final_msg_str = processed_message_bytes.decode('utf-8', errors='replace')

                        if non_ascii_replacements > 0:
                            logging.warning(f"L={target_L}, Attempt {attempt_msg_num+1}: Replaced {non_ascii_replacements} non-printable ASCII characters with '?'. Original indices: {replaced_indices}")

                        if non_ascii_replacements == len(message_bytes_for_content_check) and len(message_bytes_for_content_check) > 0:
                            logging.warning(f"L={target_L}, Attempt {attempt_msg_num+1}: Message content entirely non-printable ASCII. Discarding this attempt (CRC was ok). Avg bit confidence: {avg_confidence_this_attempt*100:.1f}%.")
                            max_avg_bit_confidence_for_this_L_call = max(max_avg_bit_confidence_for_this_L_call, avg_confidence_this_attempt)
                            # This attempt is considered failed due to content, continue K/Alpha loop
                            # (K/Alpha adjustment logic is below and will be hit if we 'continue')
                            # No 'continue' here, let it fall through to K/Alpha adjustment for the next attempt_msg_num
                        else: # Content is valid or partially replaced
                            logging.info(f"Attempt {attempt_msg_num+1} for L={target_L}: CRC VERIFIED. Message content valid (or fixed). Confidence: {avg_confidence_this_attempt*100:.1f}%.")
                            if non_ascii_replacements > 0:
                                logging.info(f"Note for L={target_L}: {non_ascii_replacements} char(s) were replaced with '?'.")
                            num_non_ascii_replacements_final = non_ascii_replacements
                            return final_msg_str, avg_confidence_this_attempt, num_non_ascii_replacements_final

                    # This case is for empty message_bytes_for_content_check (empty original message)
                    if not message_bytes_for_content_check:
                        logging.info(f"Attempt {attempt_msg_num+1} for L={target_L}: CRC VERIFIED. Empty message extracted. Confidence: {avg_confidence_this_attempt*100:.1f}%.")
                        return "", avg_confidence_this_attempt, 0


                else: # CRC Mismatch
                    logging.warning(f"Attempt {attempt_msg_num+1} for L={target_L}: CRC mismatch. Rec={received_crc_msg}, Comp={computed_crc_msg}. K={current_K_msg}, Alpha={current_alpha_estimate_msg:.2f}")
                    max_avg_bit_confidence_for_this_L_call = max(max_avg_bit_confidence_for_this_L_call, avg_confidence_this_attempt)
            except (UnicodeDecodeError, ValueError, IndexError) as e_msg_extract:
                logging.error(f"Attempt {attempt_msg_num+1} for L={target_L}: Decode/CRC error: {str(e_msg_extract)}. K={current_K_msg}, Alpha={current_alpha_estimate_msg:.2f}")
                max_avg_bit_confidence_for_this_L_call = max(max_avg_bit_confidence_for_this_L_call, avg_confidence_this_attempt)

        # If we got here, it means this attempt (for this K/Alpha) failed (CRC, bad content, or other error)
        # Adjust K/Alpha for next attempt_msg_num
        K_ADJUSTMENT_ATTEMPTS_MSG = (max_attempts_for_msg // 3) * 2 # Use local max_attempts_for_msg
        if attempt_msg_num < max_attempts_for_msg -1 : # Avoid adjustment on last failed attempt
            if attempt_msg_num < K_ADJUSTMENT_ATTEMPTS_MSG or attempt_msg_num % 3 != 0:
                increase_amount_msg = max(1, initial_K_for_msg // 2)
                current_K_msg = min(current_K_msg + increase_amount_msg, N_pool_size)
            else:
                if alpha_adjust_direction_msg == 1:
                    current_alpha_estimate_msg = min(current_alpha_estimate_msg * alpha_adjustment_factor_msg, max_alpha_estimate_msg)
                    alpha_increases_done_msg += 1
                    if alpha_increases_done_msg >= change_alpha_direction_threshold_msg or current_alpha_estimate_msg >= max_alpha_estimate_msg:
                        alpha_adjust_direction_msg = -1; alpha_decreases_done_msg = 0
                else:
                    current_alpha_estimate_msg = max(current_alpha_estimate_msg / alpha_adjustment_factor_msg, min_alpha_estimate_msg)
                    alpha_decreases_done_msg += 1
                    if alpha_decreases_done_msg >= change_alpha_direction_threshold_msg or current_alpha_estimate_msg <= min_alpha_estimate_msg:
                        alpha_adjust_direction_msg = 1; alpha_increases_done_msg = 0
            logging.info(f"Adjusting params for L={target_L} for next attempt: K={current_K_msg}, Alpha={current_alpha_estimate_msg:.2f}")

    logging.warning(f"Message extraction attempt for L={target_L} failed to verify after {max_attempts_for_msg} attempts. Max avg bit confidence over these attempts: {max_avg_bit_confidence_for_this_L_call*100:.1f}%.")
    return None, max_avg_bit_confidence_for_this_L_call, 0 # 0 replacements as no message was finalized


def extract(image_path, key, K, preprocess=False):
    """Extract a secret message from a color image with noise resilience."""
    initial_K_arg = K # Store the initial K value passed as argument
    final_message_confidence_score = -1.0
    max_confidence_from_failed_L_attempts = 0.0
    final_num_replacements = 0

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

    # Optional Pre-processing
    if preprocess:
        logging.info("Pre-processing: Enabled by user flag.")
        # _log_progress("[Progress] Phase 1: Applying image pre-processing...") # Progress log will be added in Step 8

        gray_img_for_variance = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        overall_variance = np.var(gray_img_for_variance)
        variance_threshold_for_skipping_preprocess = 1000
        logging.info(f"Overall image variance for pre-processing decision: {overall_variance:.2f}. Threshold for skipping: {variance_threshold_for_skipping_preprocess}")

        if overall_variance > variance_threshold_for_skipping_preprocess:
            logging.info("Proceeding with pre-processing operations (blur and CLAHE).")
            # Gaussian Blur
            img_processed = cv2.GaussianBlur(img, (3, 3), 1.0) # Sigma updated to 1.0
            logging.info("Pre-processing: Applied Gaussian blur (kernel=3x3, sigma=1.0).")

            # Contrast Normalization (CLAHE)
            lab = cv2.cvtColor(img_processed, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l_channel)
            limg = cv2.merge((cl, a_channel, b_channel))
            img_final_processed = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            logging.info("Pre-processing: Applied CLAHE contrast normalization (clipLimit=2.0, tileGridSize=(8,8)).")
            img = img_final_processed
        else:
            logging.info(f"Pre-processing (blur and CLAHE) skipped: Image variance ({overall_variance:.2f}) is below/equal to threshold. Original image will be used.")
            # img remains the original image
    else:
        logging.info("Pre-processing: Skipped (user flag not set).")
    
    # Pad image to multiple of 8 if necessary
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

    # Calculate Averaged DCT blocks and Averaged Block Variances
    averaged_dct_blocks = []
    averaged_block_variances = []
    for bi in range(num_blocks_i):
        for bj in range(num_blocks_j):
            sum_dct_coeffs_block = np.zeros((8, 8), dtype=np.float32)
            sum_variance_block = 0.0
            num_channels_processed_for_block = 0
            for channel_data in [b, g, r]:
                block = channel_data[bi*8:(bi+1)*8, bj*8:(bj+1)*8].astype(np.float32)
                sum_dct_coeffs_block += cv2.dct(block)
                sum_variance_block += compute_block_variance(block) # Assumes compute_block_variance takes float
                num_channels_processed_for_block += 1

            if num_channels_processed_for_block > 0: # Should always be 3 for BGR
                averaged_dct_blocks.append(sum_dct_coeffs_block / num_channels_processed_for_block)
                averaged_block_variances.append(sum_variance_block / num_channels_processed_for_block)
            else: # Should not happen with BGR images
                averaged_dct_blocks.append(sum_dct_coeffs_block) # Append zeros if no channels
                averaged_block_variances.append(0.0)


    logging.info(f"Averaged DCT and variances computed for {len(averaged_dct_blocks)} blocks.")

    # Define coefficient pool using averaged blocks (no channel index 'c')
    selected_uv = [(u, v) for u in range(1, 8) for v in range(1, 8)] # u,v from 1 to 7
    pool = [(bi, bj, u, v) for bi in range(num_blocks_i)
            for bj in range(num_blocks_j) for (u, v) in selected_uv]
    N = len(pool)
    logging.info(f"Coefficient pool size (averaged channels): {N}")

    # --- Primary Length Extraction ---
    L_final = -1
    final_message_from_extraction = None

    # Parameters for the primary length extraction loop
    current_K_len = initial_K_arg
    max_attempts_len = 15 # This is the 'max_attempts' referred to in plan
    attempt_len = 0

    current_alpha_estimate_len = 1.0
    max_alpha_estimate_len = 5.0
    min_alpha_estimate_len = 0.1
    alpha_adjustment_factor_len = 1.5
    change_alpha_direction_threshold_len = 2
    alpha_increases_done_len = 0
    alpha_decreases_done_len = 0
    alpha_adjust_direction_len = 1

    last_L_primary_attempt = -1 # Store L from the last attempt of primary loop

    logging.info(f"Starting primary length extraction. Initial K={current_K_len}, Initial AlphaEst={current_alpha_estimate_len:.2f}")
    while attempt_len < max_attempts_len:
        len_encoded_extracted = []
        desc_len = f"Extracting length (attempt {attempt_len+1}/{max_attempts_len}, K={current_K_len}, alpha_est={current_alpha_estimate_len:.2f})"
        for i in tqdm(range(28), desc=desc_len, leave=False):
            random.seed(key + i)
            idx_list = random.sample(range(N), min(current_K_len, N))
            p = [random.choice([1, -1]) for _ in range(len(idx_list))]

            sum_weighted_signal_numerator = 0.0
            sum_weights_denominator = 0.0
            bit_confidences_for_length_attempt = []

            for k_loop_idx, pool_idx in enumerate(idx_list):
                bi_pool, bj_pool, u, v = pool[pool_idx]
                block_linear_idx = bi_pool * num_blocks_j + bj_pool
                variance = averaged_block_variances[block_linear_idx]

                if variance < 1e-3: # Skip coefficients from very low-variance blocks
                    logging.debug(f"Bit {i}: Skipping coefficient from block (bi={bi_pool}, bj={bj_pool}) due to very low avg variance: {variance:.4f}")
                    continue

                adaptive_component = (1 + variance / 1000.0)
                adaptive_component = max(adaptive_component, 0.01)

                dct_coeff_val = averaged_dct_blocks[block_linear_idx][u, v]
                pattern_val = p[k_loop_idx]

                sum_weighted_signal_numerator += dct_coeff_val * pattern_val
                sum_weights_denominator += adaptive_component
                # num_coeffs_processed += 1 # If re-added for other metrics

            avg_normalized_signal = sum_weighted_signal_numerator / sum_weights_denominator if sum_weights_denominator > 0 else 0.0
            bit = 1 if avg_normalized_signal > 0.0 else 0
            len_encoded_extracted.append(bit)

            current_len_bit_confidence = min(abs(avg_normalized_signal), 1.5) / 1.5
            bit_confidences_for_length_attempt.append(current_len_bit_confidence)
            # logging.info(f"Length bit {i}, avg_norm_signal={avg_normalized_signal:.4f}, bit={bit}, K={current_K_len}, alpha_est={current_alpha_estimate_len:.2f}") # Kept K, Alpha for main log

        avg_confidence_for_length_attempt = sum(bit_confidences_for_length_attempt) / len(bit_confidences_for_length_attempt) if bit_confidences_for_length_attempt else 0.0

        # Decode length
        len_bits_extracted = []
        for j in range(0, 28, 7):
            chunk = len_encoded_extracted[j:j+7]
            len_bits_extracted.extend(hamming_decode(chunk))
        L_decoded_from_attempt = int(''.join(map(str, len_bits_extracted[:16])), 2)
        last_L_primary_attempt = L_decoded_from_attempt # Store L from this attempt
        logging.debug(f"Length extraction attempt {attempt_len+1}/{max_attempts_len}: Avg bit confidence {avg_confidence_for_length_attempt*100:.1f}%. Decoded L={L_decoded_from_attempt}")
        logging.info(f"Primary length attempt {attempt_len+1}: Decoded L_attempt={L_decoded_from_attempt}. K={current_K_len}, Alpha={current_alpha_estimate_len:.2f}")


        if L_decoded_from_attempt > 0 and L_decoded_from_attempt <= 1024 and L_decoded_from_attempt % 128 == 0:
            L_final = L_decoded_from_attempt
            logging.info(f"Primary length extraction successful on attempt {attempt_len+1}. Valid L={L_final} found.")
            break

        attempt_len += 1
        if attempt_len < max_attempts_len: # Only adjust if not the last attempt
            logging.info(f"Primary length attempt {attempt_len} failed to yield compliant L. Current K={current_K_len}, AlphaEst={current_alpha_estimate_len:.2f}")
            K_ADJUSTMENT_ATTEMPTS_LEN = (max_attempts_len // 3) * 2
            if attempt_len < K_ADJUSTMENT_ATTEMPTS_LEN or attempt_len % 3 != 0:
                increase_amount_len = max(1, initial_K_arg // 2)
                current_K_len = min(current_K_len + increase_amount_len, N)
            else:
                if alpha_adjust_direction_len == 1:
                    current_alpha_estimate_len = min(current_alpha_estimate_len * alpha_adjustment_factor_len, max_alpha_estimate_len)
                    alpha_increases_done_len += 1
                    if alpha_increases_done_len >= change_alpha_direction_threshold_len or current_alpha_estimate_len >= max_alpha_estimate_len:
                        alpha_adjust_direction_len = -1; alpha_decreases_done_len = 0
                else:
                    current_alpha_estimate_len = max(current_alpha_estimate_len / alpha_adjustment_factor_len, min_alpha_estimate_len)
                    alpha_decreases_done_len += 1
                    if alpha_decreases_done_len >= change_alpha_direction_threshold_len or current_alpha_estimate_len <= min_alpha_estimate_len:
                        alpha_adjust_direction_len = 1; alpha_increases_done_len = 0
            logging.info(f"Adjusting parameters for next length attempt: new K={current_K_len}, new alpha_est={current_alpha_estimate_len:.2f}")

    # Store final K and Alpha from primary length loop for potential error message
    final_K_primary_len_loop = current_K_len
    final_alpha_primary_len_loop = current_alpha_estimate_len

    # === Fallback Length Logic ===
    if L_final == -1: # If primary length extraction failed
        logging.warning(f"Primary length extraction failed after {max_attempts_len} attempts (last L_attempt={last_L_primary_attempt}). Initiating fallback length search.")

        fallback_L_candidates = []
        if last_L_primary_attempt > 0: # Only if last_L_primary_attempt is somewhat sensible
            L_candidate1 = (last_L_primary_attempt // 128) * 128
            L_candidate2 = ((last_L_primary_attempt // 128) + 1) * 128
            if L_candidate1 > 0 and L_candidate1 <= 1024:
                fallback_L_candidates.append(L_candidate1)
            if L_candidate2 > 0 and L_candidate2 <= 1024 and L_candidate2 != L_candidate1:
                fallback_L_candidates.append(L_candidate2)

        # Add fixed common lengths if not already covered, ensuring they are valid
        common_lengths = [128, 256, 384, 512] # Common AES block multiples
        for cl in common_lengths:
            if cl <= 1024 and cl not in fallback_L_candidates:
                 fallback_L_candidates.append(cl)
        fallback_L_candidates.sort()


        if not fallback_L_candidates:
            logging.warning("No valid fallback L candidates generated.")
        else:
            logging.info(f"Fallback L candidates: {fallback_L_candidates}")

        fallback_max_attempts = max(1, max_attempts_len // 3) # Reduced attempts for fallback

        for candidate_L in fallback_L_candidates:
            logging.info(f"Fallback: Attempting message extraction with L_candidate={candidate_L}.")
            # Call helper for message extraction
            result_message, confidence_from_call, replacements_in_call = _extract_message_for_length(
                candidate_L, initial_K_arg, fallback_max_attempts, key, N, pool,
                num_blocks_j, averaged_block_variances, averaged_dct_blocks, max_attempts_len
            )
            if result_message is not None:
                L_final = candidate_L
                final_message_from_extraction = result_message
                final_message_confidence_score = confidence_from_call
                final_num_replacements = replacements_in_call
                logging.info(f"Fallback successful: Valid message extracted with L={L_final}. Confidence: {final_message_confidence_score*100:.1f}%. Replacements: {final_num_replacements}.")
                break
            else:
                max_confidence_from_failed_L_attempts = max(max_confidence_from_failed_L_attempts, confidence_from_call)
                logging.warning(f"Fallback attempt with L_candidate={candidate_L} failed. Confidence in this attempt: {confidence_from_call*100:.1f}%.")

        if L_final == -1:
            error_msg = (f"Failed to extract a valid message length after {max_attempts_len} primary attempts and "
                         f"{len(fallback_L_candidates)} fallback(s) (last primary K={final_K_primary_len_loop}, "
                         f"alpha={final_alpha_primary_len_loop:.2f}). Max signal confidence from failed L attempts "
                         f"({max_confidence_from_failed_L_attempts*100:.1f}%), suggests the image may not contain a recognizable "
                         f"watermark or the key is incorrect. Possible causes: incorrect key, severe image "
                         f"compression/noise, or no watermark present. Consider using --preprocess flag if image is noisy/compressed.")
            logging.error(error_msg)
            raise ValueError(error_msg)

    if L_final == -1:
        error_msg_fatal = "Fatal error in length extraction: No valid L found after all primary and fallback attempts."
        logging.error(error_msg_fatal)
        raise ValueError(error_msg_fatal)

    if final_message_from_extraction is None:
        logging.info(f"Proceeding to main message extraction with L={L_final} (determined from primary attempts).")
        result_message, confidence_from_call, replacements_in_call = _extract_message_for_length(
            L_final, initial_K_arg, max_attempts_len, key, N, pool,
            num_blocks_j, averaged_block_variances, averaged_dct_blocks, max_attempts_len
        )
        if result_message is not None:
            final_message_from_extraction = result_message
            final_message_confidence_score = confidence_from_call
            final_num_replacements = replacements_in_call
        else:
            max_confidence_from_failed_L_attempts = max(max_confidence_from_failed_L_attempts, confidence_from_call)
            error_msg = (f"Message extraction failed for L={L_final} after all attempts (max {max_attempts_len} attempts per L). "
                         f"Max signal confidence from these failed attempts: {max_confidence_from_failed_L_attempts*100:.1f}%. "
                         f"Possible causes: incorrect key, image corruption, or high noise levels affecting message bits. "
                         f"Try --preprocess or verify image integrity. Check logs for K/Alpha values used in the final attempts for this L.")
            logging.error(error_msg)
            raise ValueError(error_msg)

    if final_message_from_extraction is not None:
        logging.info(f"Message extracted successfully. Confidence: {final_message_confidence_score*100:.1f}%. Replacements: {final_num_replacements}. Message: '{final_message_from_extraction}'")

    return final_message_from_extraction, final_message_confidence_score, final_num_replacements


# (The old message extraction loop is now removed as its logic is in _extract_message_for_length)
# ... The rest of the file (if __name__ == "__main__": block) remains the same ...

# Remove the old message extraction loop from here down to its error handling
# This was the original main message extraction loop:
#
#    # Extract message bits
#    num_chunks = (L + 3) // 4
#    M_msg = num_chunks * 7
#    msg_encoded_extracted = []
#    attempt = 0 # Reset attempt counter for message extraction
#    # Reset K to initial K for message extraction, alpha parameters are also reset
#    current_K = initial_K_arg # Use the stored initial K
#
#    # Initialize Adaptive Parameters for message extraction
#    current_alpha_estimate = 1.0
#    # ... (rest of the original message extraction loop, which is now in the helper) ...
#    # ... up to ...
#    if attempt >= max_attempts:
#        logging.error(f"Failed to extract valid message after {max_attempts} attempts (L={L}). Final K={current_K}, alpha_est={current_alpha_estimate:.2f}")
#        raise ValueError(f"Failed to extract valid message after {max_attempts} attempts (L={L}). Max K reached: {current_K}, final alpha estimate: {current_alpha_estimate:.2f}. CRC or decoding failed. Ensure key is correct and image integrity.")
#
#    return msg_extracted
#
# This entire block needs to be replaced by the logic calling the helper function.
# The diff will show this removal implicitly by replacing the section.

def extract(image_path, key, K, preprocess=False): # Original start of function to be replaced by the new one above.
    initial_K_arg = K # Store the initial K value passed as argument
            p = [random.choice([1, -1]) for _ in range(len(idx_list))]

            sum_weighted_signal_numerator = 0.0
            sum_weights_denominator = 0.0
    # This is the starting point of the original extract function body
    # The SEARCH block above will capture the old message extraction loop
    # The REPLACE block for the extract function will contain the new logic,
    # including calling the _extract_message_for_length helper.
    # The old message extraction loop content itself is not needed here in the SEARCH
    # as the entire function body from initial_K_arg down to its return is being replaced.
    # This SEARCH block is minimal just to provide context for the start of the function.
    initial_K_arg = K

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
    embed_parser.add_argument("--preprocess", action="store_true", help="Placeholder for pre-processing flag (currently no action in embed).")

    # Extract command
    extract_parser = subparsers.add_parser("extract", help="Extract a message from a color image")
    extract_parser.add_argument("--image", required=True, help="Input watermarked color image path (PNG/JPEG)")
    extract_parser.add_argument("--key", type=int, required=True, help="Key used during embedding")
    extract_parser.add_argument("--K", type=int, default=500, help="Initial coefficients per bit (default: 500)")
    extract_parser.add_argument("--preprocess", action="store_true", help="Enable pre-processing (Gaussian blur, contrast normalization) on the input image before extraction.")

    args = parser.parse_args()

    try:
        if args.command == "embed":
            embed(args.image, args.message, args.output, args.key, args.alpha, args.K, preprocess=args.preprocess)
            print(f"Message embedded successfully into {args.output}")
        elif args.command == "extract":
            extracted_data = extract(args.image, args.key, args.K, preprocess=args.preprocess)
            if extracted_data : # Check if not None, in case of future direct None returns from extract
                message, confidence, num_replacements = extracted_data
                if message is not None:
                    print_msg = f"Extracted message: {message}, Confidence: {confidence*100:.1f}%"
                    if num_replacements > 0:
                        print_msg += f" (Note: {num_replacements} non-ASCII char(s) replaced with '?')"
                    print(print_msg)
                else:
                    # This path should be rare as extract() is designed to raise ValueErrors on failure
                    print(f"Extraction failed to retrieve a message. Max observed confidence during failed attempts: {confidence*100:.1f}%.")
            else:
                 # This path implies extract() returned None, which it currently doesn't (it raises errors).
                 # Adding for robustness in case of future changes to extract's error handling.
                print("Extraction process did not yield a result or failed unexpectedly before raising a specific error.")
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        print(f"Error: {str(e)}")