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
    # Remove padding more safely
    if not padded_message: # Handle case of empty padded_message
        logging.error("Decryption error: Padded message is empty.")
        raise ValueError("Decryption error: Padded message is empty.")

    pad_length = padded_message[-1]
    # Validate pad_length: must be between 1 and 16 (or block size of AES)
    # and not greater than the length of the padded message itself.
    if pad_length == 0 or pad_length > 16 or pad_length > len(padded_message):
        logging.error(f"Decryption error: Invalid padding length {pad_length}. Padded message length: {len(padded_message)}")
        # Potentially log part of the message for debugging if allowed by security policy, e.g., padded_message[:10]
        raise ValueError("Decryption error: Invalid padding.")

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

def _extract_message_for_length(
    target_L,
    initial_K_for_msg,
    max_attempts_for_msg,
    key_param,
    N_pool_size,
    pool_details, # List of (bi, bj, u, v) for averaged blocks
    num_blocks_j_img,
    raw_dct_blocks_input, # Renamed from averaged_dct_coefficients
    raw_block_variances_input, # Renamed from averaged_block_variances_input
    active_channel_weights, # New parameter: list e.g. [wb, wg, wr]
    passed_initial_alpha_estimate # Alpha estimate from length phase or default, used for adaptive alpha and confidence scaling
):
    logging.info(f"Message Extraction Helper for L={target_L} using channel weights: B={active_channel_weights[0]:.3f}, G={active_channel_weights[1]:.3f}, R={active_channel_weights[2]:.3f}")

    # Define overall_estimated_alpha for confidence scaling, ensuring it's not too small
    overall_estimated_alpha = max(0.01, passed_initial_alpha_estimate)
    logging.info(f"Confidence scaling for L={target_L} will use overall_estimated_alpha: {overall_estimated_alpha:.3f} (derived from passed_initial_alpha_estimate: {passed_initial_alpha_estimate:.3f})")

    max_avg_bit_confidence_for_this_L_call = 0.0
    num_non_ascii_replacements_final = 0
    final_K_for_this_L_call = initial_K_for_msg # Store the K that led to success/failure for this L
    final_alpha_for_this_L_call = passed_initial_alpha_estimate # Store alpha for this L

    current_K_msg = initial_K_for_msg
    current_alpha_estimate_msg = passed_initial_alpha_estimate # Use passed alpha
    max_alpha_estimate_msg = 5.0
    min_alpha_estimate_msg = 0.1
    alpha_adjustment_factor_msg = 1.5
    change_alpha_direction_threshold_msg = 2 # How many times to inc/dec alpha before changing direction
    alpha_increases_done_msg = 0
    alpha_decreases_done_msg = 0
    alpha_adjust_direction_msg = 1 # 1 for increase, -1 for decrease

    logging.info(f"Message Extraction Helper: Attempting L={target_L}, initial_K={current_K_msg}, initial_AlphaEst={current_alpha_estimate_msg:.2f}, max_attempts={max_attempts_for_msg}")

    for attempt_msg_num in range(max_attempts_for_msg):
        final_K_for_this_L_call = current_K_msg # Update K for each attempt
        final_alpha_for_this_L_call = current_alpha_estimate_msg # Update Alpha for each attempt

        msg_encoded_extracted = []
        bit_confidences_this_attempt = []
        num_chunks_msg = (target_L + 3) // 4 # Ensure integer division for padding
        M_bits_to_extract_msg = num_chunks_msg * 7

        desc_msg = f"Extracting L={target_L} (attempt {attempt_msg_num+1}/{max_attempts_for_msg}, K={current_K_msg}, alpha_est={current_alpha_estimate_msg:.2f})"
        for i_msg_bit_loop in tqdm(range(28, 28 + M_bits_to_extract_msg), desc=desc_msg, leave=False):
            random.seed(key_param + i_msg_bit_loop) # Seed includes main key and bit index
            idx_list_msg = random.sample(range(N_pool_size), min(current_K_msg, N_pool_size))
            # --- Start of Python code block for subtask point 3 ---
            if idx_list_msg: # Check if list is not empty for safety
                _selected_coeff_float_values = []
                for _p_idx in idx_list_msg:
                    _pool_item = pool_details[_p_idx]
                    _bi, _bj, _c, _u, _v = _pool_item
                    _block_idx = _bi * num_blocks_j_img + _bj
                    _coeff = raw_dct_blocks_input[_c][_block_idx][_u, _v]
                    _selected_coeff_float_values.append(_coeff)

                if _selected_coeff_float_values: # If list was successfully populated
                    _coeff_magnitudes = np.abs(np.array(_selected_coeff_float_values))
                    _near_zero_count = np.sum(_coeff_magnitudes < 0.01)
                    _avg_magnitude = np.mean(_coeff_magnitudes) # np.mean handles empty array by returning nan, but we check _selected_coeff_float_values first.
                    logging.debug(f"Bit {i_msg_bit_loop-28} (L={target_L}, K={current_K_msg}): Selected {len(idx_list_msg)} coeffs stats - NearZeroCount: {_near_zero_count}, AvgAbsMag: {_avg_magnitude:.4f}")
                else:
                    logging.debug(f"Bit {i_msg_bit_loop-28} (L={target_L}, K={current_K_msg}): idx_list_msg was non-empty, but failed to populate coeffs for stats.")
            else:
                logging.debug(f"Bit {i_msg_bit_loop-28} (L={target_L}, K={current_K_msg}): idx_list_msg was empty, no coeff stats to log.")
            # --- End of Python code block for subtask point 3 ---
            p_msg = [random.choice([1, -1]) for _ in range(len(idx_list_msg))]

            sum_corr_numerator_msg = 0.0 # sum(dct_coeff * pattern)
            sum_adaptive_denominator_msg = 0.0 # sum(adaptive_component) for normalization

            if not idx_list_msg and current_K_msg > 0 : # Should not happen if N_pool_size > 0
                logging.debug(f"Bit {i_msg_bit_loop-28} (L={target_L}): Empty idx_list_msg with K={current_K_msg}. N_pool_size={N_pool_size}")
                # This would lead to avg_normalized_signal_msg = 0, bit = 0.

            for k_loop_idx_msg, pool_idx_msg in enumerate(idx_list_msg):
                bi_pool, bj_pool, c_pool, u, v = pool_details[pool_idx_msg] # Now includes c_pool
                block_linear_idx_msg = bi_pool * num_blocks_j_img + bj_pool

                dct_coeff_val_msg = raw_dct_blocks_input[c_pool][block_linear_idx_msg][u, v]
                variance_msg = raw_block_variances_input[c_pool][block_linear_idx_msg]

                # Adaptive component calculation (consistent with length part)
                adaptive_component_msg = max((1 + variance_msg / 1000.0), 0.05) # From original code

                sum_corr_numerator_msg += dct_coeff_val_msg * p_msg[k_loop_idx_msg] * active_channel_weights[c_pool]
                sum_adaptive_denominator_msg += adaptive_component_msg

            # Normalize the signal sum by the sum of adaptive components
            # This calculation is for the overall confidence score, using all current_K_msg coefficients
            avg_normalized_signal_msg = sum_corr_numerator_msg / sum_adaptive_denominator_msg if sum_adaptive_denominator_msg > 1e-9 else 0.0

            # Majority Voting for bit extraction with Confidence Weighting
            votes_with_confidence = [] # Stores tuples of (vote_bit, vote_confidence)
            num_sub_samples = 3
            K_sub_sample_base = current_K_msg // 2

            for sub_sample_idx in range(num_sub_samples):
                actual_K_for_sub_sample = max(1, K_sub_sample_base)
                if not idx_list_msg:
                    logging.warning(f"Bit {i_msg_bit_loop-28} (L={target_L}), SubSample {sub_sample_idx}: idx_list_msg is empty. Cannot vote.")
                    votes_with_confidence.append((0, 0.0)) # Default vote with zero confidence
                    continue

                sub_sample_indices = random.sample(idx_list_msg, min(actual_K_for_sub_sample, len(idx_list_msg)))

                sum_corr_sub_sample = 0.0
                sum_adaptive_sub_sample = 0.0
                p_sub_mapped = []

                for sub_idx_val in sub_sample_indices:
                    try:
                        original_pos_in_idx_list_msg = idx_list_msg.index(sub_idx_val)
                        p_sub_mapped.append(p_msg[original_pos_in_idx_list_msg])
                    except ValueError:
                        logging.error(f"Error mapping sub-sample index for p_val. Bit {i_msg_bit_loop-28}, SubSample {sub_sample_idx}")
                        continue

                if len(p_sub_mapped) != len(sub_sample_indices): # If any mapping error occurred
                    logging.warning(f"Bit {i_msg_bit_loop-28} (L={target_L}), SubSample {sub_sample_idx}: p_val mapping issue. Cannot vote reliably.")
                    votes_with_confidence.append((0, 0.0))
                    continue

                if not sub_sample_indices:
                    logging.debug(f"Bit {i_msg_bit_loop-28} (L={target_L}), SubSample {sub_sample_idx}: sub_sample_indices is empty. Voting 0 with 0 confidence.")
                    votes_with_confidence.append((0, 0.0))
                    continue

                for k_sub_idx, pool_idx_sub_sample in enumerate(sub_sample_indices):
                    bi_pool, bj_pool, c_pool_sub, u, v = pool_details[pool_idx_sub_sample]
                    block_linear_idx_sub_sample = bi_pool * num_blocks_j_img + bj_pool

                    dct_coeff_val_sub = raw_dct_blocks_input[c_pool_sub][block_linear_idx_sub_sample][u, v]
                    variance_sub = raw_block_variances_input[c_pool_sub][block_linear_idx_sub_sample]
                    adaptive_component_sub = max((1 + variance_sub / 1000.0), 0.05)

                    sum_corr_sub_sample += dct_coeff_val_sub * p_sub_mapped[k_sub_idx] * active_channel_weights[c_pool_sub]
                    sum_adaptive_sub_sample += adaptive_component_sub

                avg_normalized_signal_sub_sample = sum_corr_sub_sample / sum_adaptive_sub_sample if sum_adaptive_sub_sample > 1e-9 else 0.0
                current_vote_bit = 1 if avg_normalized_signal_sub_sample > 0.0 else 0
                votes_with_confidence.append((current_vote_bit, avg_normalized_signal_sub_sample))

            # Confidence-Weighted Voting
            total_weighted_sum = 0.0
            for vote_bit, vote_confidence_signal in votes_with_confidence:
                bipolar_vote = 1 if vote_bit == 1 else -1
                total_weighted_sum += bipolar_vote * abs(vote_confidence_signal) # Weight by magnitude of confidence signal

            final_bit_value = 1 if total_weighted_sum > 0.0 else 0
            # Handle exact zero sum case (e.g. if all confidences are zero, or perfect tie with symmetrical confidences)
            if total_weighted_sum == 0.0:
                # Default to 0 or use a tie-breaking rule, e.g., first vote, or majority if confidences were equal.
                # For simplicity, defaulting to 0.
                final_bit_value = 0
                logging.debug(f"Bit {i_msg_bit_loop-28} (L={target_L}): Weighted sum is zero, defaulting bit to 0.")

            formatted_votes_conf = [f"({v},{c:.3f})" for v, c in votes_with_confidence]
            logging.info(f"Bit {i_msg_bit_loop-28} (L={target_L}): VotesWithConf=[{', '.join(formatted_votes_conf)}], WeightedSum={total_weighted_sum:.3f}, SelectedBit={final_bit_value}")
            msg_encoded_extracted.append(final_bit_value)

            # Calculate confidence for this bit (0 to 1 range)
            # avg_normalized_signal_msg is an estimate of the actual alpha used during embedding for this bit's extraction (after channel weighting)
            current_bit_confidence = min(1.0, abs(avg_normalized_signal_msg) / overall_estimated_alpha)
            bit_confidences_this_attempt.append(current_bit_confidence)

        avg_confidence_this_attempt = sum(bit_confidences_this_attempt) / len(bit_confidences_this_attempt) if bit_confidences_this_attempt else 0.0
        max_avg_bit_confidence_for_this_L_call = max(max_avg_bit_confidence_for_this_L_call, avg_confidence_this_attempt)
        logging.debug(f"Attempt {attempt_msg_num+1} for L={target_L}: Avg bit confidence {avg_confidence_this_attempt*100:.1f}% (K={current_K_msg}, AlphaEst={current_alpha_estimate_msg:.2f})")

        msg_bits_extracted = []
        if len(msg_encoded_extracted) != M_bits_to_extract_msg:
             logging.warning(f"Attempt {attempt_msg_num+1} for L={target_L}: Encoded bits length {len(msg_encoded_extracted)} != expected {M_bits_to_extract_msg}. Skipping CRC.")
        else:
            for j_msg in range(0, len(msg_encoded_extracted), 7):
                chunk_msg = msg_encoded_extracted[j_msg:j_msg+7]
                msg_bits_extracted.extend(hamming_decode(chunk_msg))
            msg_bits_extracted = msg_bits_extracted[:target_L] # Trim to target_L

        if len(msg_bits_extracted) != target_L:
            logging.warning(f"Attempt {attempt_msg_num+1} for L={target_L}: Decoded bits length {len(msg_bits_extracted)} != target {target_L}. (K={current_K_msg}, AlphaEst={current_alpha_estimate_msg:.2f})")
        else:
            extracted_bytes_msg = bytes([int(''.join(map(str, msg_bits_extracted[k_byte:k_byte+8])), 2) for k_byte in range(0, target_L, 8)])
            try:
                decrypted_bytes_msg = aes_decrypt(extracted_bytes_msg, key_param)
                if len(decrypted_bytes_msg) < 4: # Must be at least 4 bytes for CRC
                    raise ValueError("Decrypted data too short for CRC.")

                received_crc_msg = int.from_bytes(decrypted_bytes_msg[-4:], 'big')
                message_bytes_for_content_check = decrypted_bytes_msg[:-4] # Exclude CRC for content check
                computed_crc_msg = compute_crc32(message_bytes_for_content_check)

                if received_crc_msg == computed_crc_msg:
                    processed_message_bytes = bytearray()
                    non_ascii_replacements = 0

                    if not message_bytes_for_content_check: # Empty message after CRC removal
                        final_msg_str = ""
                    else:
                        for byte_val in message_bytes_for_content_check:
                            if not (32 <= byte_val <= 126): # Check for printable ASCII
                                processed_message_bytes.append(ord('?')) # Replace non-printable
                                non_ascii_replacements += 1
                            else:
                                processed_message_bytes.append(byte_val)
                        final_msg_str = processed_message_bytes.decode('ascii', errors='replace') # Ensure ASCII

                        if non_ascii_replacements > 0:
                            logging.info(f"L={target_L}, Attempt {attempt_msg_num+1}: Replaced {non_ascii_replacements} non-printable/non-ASCII characters with '?'.")

                        # Check if the entire message was replaced (all non-printable)
                        if non_ascii_replacements == len(message_bytes_for_content_check) and len(message_bytes_for_content_check) > 0:
                            logging.warning(f"L={target_L}, Attempt {attempt_msg_num+1}: Message content entirely non-printable/non-ASCII. Discarding (CRC OK). Avg bit conf: {avg_confidence_this_attempt*100:.1f}%.")
                            # Continue to next attempt (K/Alpha adjustment)
                        else: # Content is valid or partially replaced and acceptable
                            logging.info(f"Attempt {attempt_msg_num+1} for L={target_L}: CRC VERIFIED. Content OK. Confidence: {avg_confidence_this_attempt*100:.1f}%. K={current_K_msg}, AlphaEst={current_alpha_estimate_msg:.2f}")
                            num_non_ascii_replacements_final = non_ascii_replacements
                            return final_msg_str, avg_confidence_this_attempt, num_non_ascii_replacements_final, current_K_msg, current_alpha_estimate_msg

                    # Handle empty original message (CRC was still checked on empty bytes + CRC bytes)
                    if not message_bytes_for_content_check: # If the original message was empty
                        logging.info(f"Attempt {attempt_msg_num+1} for L={target_L}: CRC VERIFIED. Empty message. Confidence: {avg_confidence_this_attempt*100:.1f}%. K={current_K_msg}, AlphaEst={current_alpha_estimate_msg:.2f}")
                        return "", avg_confidence_this_attempt, 0, current_K_msg, current_alpha_estimate_msg # Return empty string, 0 replacements
                else: # CRC Mismatch
                    logging.warning(f"Attempt {attempt_msg_num+1} for L={target_L}: CRC mismatch. Rec={received_crc_msg}, Comp={computed_crc_msg}. K={current_K_msg}, AlphaEst={current_alpha_estimate_msg:.2f}")
            except ValueError as e_msg_extract: # Catch specific errors like padding or short data
                logging.error(f"Attempt {attempt_msg_num+1} for L={target_L}: AES Decrypt/CRC/Padding error: {str(e_msg_extract)}. K={current_K_msg}, AlphaEst={current_alpha_estimate_msg:.2f}")
            except Exception as e_generic: # Catch any other unexpected errors during decryption/CRC
                 logging.error(f"Attempt {attempt_msg_num+1} for L={target_L}: Unexpected error in processing: {str(e_generic)}. K={current_K_msg}, AlphaEst={current_alpha_estimate_msg:.2f}")

        # K/Alpha Adjustment Logic for next attempt (if CRC failed or content was bad)
        if attempt_msg_num < max_attempts_for_msg -1 : # Avoid adjustment on last failed attempt
            # Strategy: First half of attempts, primarily adjust K. Second half, adjust Alpha, with small K nudges.
            if attempt_msg_num < max_attempts_for_msg // 2 :
                increase_amount_msg = max(1, initial_K_for_msg // 5) # Increase K by a fraction of initial K
                current_K_msg = min(current_K_msg + increase_amount_msg, N_pool_size) # Cap K at N
                logging.info(f"Adjusting params for L={target_L} for next attempt: New K={current_K_msg} (AlphaEst={current_alpha_estimate_msg:.2f})")
            else: # Second half of attempts, focus on Alpha, with some K nudging
                # Adjust Alpha
                if alpha_adjust_direction_msg == 1:
                    current_alpha_estimate_msg = min(current_alpha_estimate_msg * alpha_adjustment_factor_msg, max_alpha_estimate_msg)
                    alpha_increases_done_msg += 1
                    if alpha_increases_done_msg >= change_alpha_direction_threshold_msg or current_alpha_estimate_msg >= max_alpha_estimate_msg :
                        alpha_adjust_direction_msg = -1; alpha_decreases_done_msg = 0 # Change direction
                else: # direction is -1 (decrease alpha)
                    current_alpha_estimate_msg = max(current_alpha_estimate_msg / alpha_adjustment_factor_msg, min_alpha_estimate_msg)
                    alpha_decreases_done_msg += 1
                    if alpha_decreases_done_msg >= change_alpha_direction_threshold_msg or current_alpha_estimate_msg <= min_alpha_estimate_msg:
                        alpha_adjust_direction_msg = 1; alpha_increases_done_msg = 0 # Change direction
                logging.info(f"Adjusting params for L={target_L} for next attempt: New AlphaEst={current_alpha_estimate_msg:.2f} (K={current_K_msg})")

                # Occasional K nudge during alpha phase
                if attempt_msg_num % 3 == 0 and attempt_msg_num > max_attempts_for_msg // 2 : # Every 3 attempts in the alpha phase
                    k_nudge = initial_K_for_msg // 10
                    if alpha_adjust_direction_msg == 1 : # If alpha is generally increasing, maybe K can be slightly lower
                        current_K_msg = max(initial_K_for_msg // 2, current_K_msg - k_nudge)
                    else: # If alpha is generally decreasing, maybe K needs to be higher
                         current_K_msg = min(N_pool_size, current_K_msg + k_nudge)


    logging.warning(f"Message extraction helper for L={target_L} failed after {max_attempts_for_msg} attempts. Max avg bit confidence: {max_avg_bit_confidence_for_this_L_call*100:.1f}%. Final K={final_K_for_this_L_call}, Final AlphaEst={final_alpha_for_this_L_call:.2f}")
    return None, max_avg_bit_confidence_for_this_L_call, 0, final_K_for_this_L_call, final_alpha_for_this_L_call


def extract(image_path, key, K, preprocess=False, blur_sigma=0.7, clahe_clip_limit=1.5, variance_threshold=1000.0):
    initial_K_arg = K
    final_extracted_message = None
    final_message_confidence_score = 0.0 # Default to 0
    final_num_replacements = 0

    # For summary log
    final_L_value = -1
    k_at_L_detection = initial_K_arg
    alpha_at_L_detection = 1.0
    k_at_msg_extraction = initial_K_arg
    alpha_at_msg_extraction = 1.0

    actionable_error_message_parts = [] # Collect parts for a comprehensive error

    valid_extensions = ['.png', '.jpg', '.jpeg']
    if not any(image_path.lower().endswith(ext) for ext in valid_extensions):
        err_msg = f"Invalid image format: {image_path}. Must be PNG or JPEG."
        logging.error(err_msg)
        raise ValueError(err_msg)
    
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        err_msg = f"Cannot load image at {image_path}. File not found or corrupted."
        logging.error(err_msg)
        raise FileNotFoundError(err_msg)
    if len(img.shape) != 3 or img.shape[2] != 3:
        err_msg = "Input image must be a color image with 3 channels (e.g., BGR)."
        logging.error(err_msg)
        raise ValueError(err_msg)

    preprocess_actually_applied = False # For summary log
    if preprocess:
        logging.info("Pre-processing: Enabled by user flag.")
        gray_img_for_variance = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        overall_variance = np.var(gray_img_for_variance)
        variance_threshold_for_skipping_preprocess = variance_threshold
        logging.info(f"Overall image variance for pre-processing decision: {overall_variance:.2f}. Threshold: {variance_threshold_for_skipping_preprocess:.2f}")

        if overall_variance > variance_threshold_for_skipping_preprocess:
            logging.info(f"Proceeding with pre-processing operations (blur and CLAHE) as overall variance {overall_variance:.2f} > threshold {variance_threshold_for_skipping_preprocess:.2f}.")
            sigma_blur = blur_sigma
            img_blurred = cv2.GaussianBlur(img, (3, 3), sigma_blur)
            logging.info(f"Pre-processing: Applied Gaussian blur (kernel=3x3, sigma={sigma_blur:.2f}).")

            clip_limit_clahe = clahe_clip_limit
            lab = cv2.cvtColor(img_blurred, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=clip_limit_clahe, tileGridSize=(8, 8))
            cl = clahe.apply(l_channel)
            limg = cv2.merge((cl, a_channel, b_channel))
            img_processed_final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            logging.info(f"Pre-processing: Applied CLAHE contrast normalization (clipLimit={clip_limit_clahe:.2f}, tileGridSize=(8,8)).")
            img = img_processed_final
            preprocess_actually_applied = True
        else:
            logging.info(f"Pre-processing (blur and CLAHE) skipped: Image variance ({overall_variance:.2f}) is not > threshold ({variance_threshold_for_skipping_preprocess:.2f}). Original image used.")
    else:
        logging.info("Pre-processing: Skipped (user flag not set).")
    actionable_error_message_parts.append(f"Pre-processed={preprocess_actually_applied}")
    if preprocess_actually_applied:
        actionable_error_message_parts.append(f"BlurSigmaUsed={blur_sigma:.2f}")
        actionable_error_message_parts.append(f"CLAHELimitUsed={clahe_clip_limit:.2f}")
    
    # Pad image to multiple of 8 if necessary
    h, w, _ = img.shape
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    if pad_h > 0 or pad_w > 0:
        img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)
        h, w, _ = img.shape
    logging.info(f"Image dimensions after padding (if any): {h}x{w}, blocks: {h//8}x{w//8}")

    # Split into channels
    b_ch, g_ch, r_ch = cv2.split(img) # Renamed for clarity

    num_blocks_i, num_blocks_j = h // 8, w // 8
    num_blocks_total = num_blocks_i * num_blocks_j

    if num_blocks_total == 0:
        err_msg = "Image is too small, results in zero 8x8 blocks after padding."
        logging.error(err_msg)
        # Summary log already handled by the final try-except block's finally clause
        raise ValueError(err_msg)

    # Calculate Per-Channel DCT Strengths
    channel_total_abs_dct_sum = {'B': 0.0, 'G': 0.0, 'R': 0.0}
    # Store raw DCTs and variances per channel for later averaging
    raw_dct_blocks_per_channel = [[np.zeros((8,8), dtype=np.float32) for _ in range(num_blocks_total)] for _ in range(3)]
    block_variances_per_channel = [[0.0 for _ in range(num_blocks_total)] for _ in range(3)]

    for c_idx, channel_data in enumerate([b_ch, g_ch, r_ch]):
        channel_key = ['B', 'G', 'R'][c_idx]
        for bi in range(num_blocks_i):
            for bj in range(num_blocks_j):
                block_idx_flat = bi * num_blocks_j + bj
                block = channel_data[bi*8:(bi+1)*8, bj*8:(bj+1)*8].astype(np.float32)

                dct_coeffs_block = cv2.dct(block)
                raw_dct_blocks_per_channel[c_idx][block_idx_flat] = dct_coeffs_block
                block_variances_per_channel[c_idx][block_idx_flat] = compute_block_variance(block)

                # Sum of absolute values of mid-frequency DCT coefficients (excluding DC at [0,0])
                block_strength = np.sum(np.abs(dct_coeffs_block[1:, 1:]))
                channel_total_abs_dct_sum[channel_key] += block_strength

    # Normalize and Log Channel Weights
    total_strength_all_channels = sum(channel_total_abs_dct_sum.values())
    channel_weights = {'B': 1/3, 'G': 1/3, 'R': 1/3} # Default to equal weights

    if total_strength_all_channels > 1e-9: # Avoid division by zero
        channel_weights = {ch: strength / total_strength_all_channels
                           for ch, strength in channel_total_abs_dct_sum.items()}

    logging.info(f"Calculated channel weights (for info, current averaging is equal): "
                 f"B={channel_weights['B']:.3f}, G={channel_weights['G']:.3f}, R={channel_weights['R']:.3f}")

    # DCT Coefficient Sparsity Analysis (Task 9 Rec. 1)
    # Note: selected_uv is defined later. For this analysis, we'll use a fixed range or ensure
    # it's defined if this block were to be moved after selected_uv's main definition.
    # Given current placement, a temporary or fixed selected_uv is needed here.
    # The prompt specifies using (1,6) for temp_selected_uv_for_sparsity.
    if raw_dct_blocks_per_channel: # Ensure variables are populated
        sparsity_metrics_per_channel = []
        # Using a fixed range as specified, similar to what selected_uv will become.
        temp_selected_uv_for_sparsity = [(u, v) for u in range(1, 6) for v in range(1, 6)]

        total_coeffs_count = 0
        near_zero_coeffs_count = 0

        for c_idx in range(len(raw_dct_blocks_per_channel)):
            channel_coeffs_count = 0
            channel_near_zero_coeffs = 0
            if not raw_dct_blocks_per_channel[c_idx]: # Check if channel data exists
                logging.warning(f"Sparsity: Channel {c_idx} has no DCT blocks.")
                continue

            for block_dct_coeffs in raw_dct_blocks_per_channel[c_idx]:
                if block_dct_coeffs is None or not hasattr(block_dct_coeffs, 'shape'): # Additional check for block validity
                    logging.warning(f"Sparsity: Invalid/empty DCT block encountered in channel {c_idx}.")
                    continue
                for u, v in temp_selected_uv_for_sparsity:
                    if u < block_dct_coeffs.shape[0] and v < block_dct_coeffs.shape[1]:
                        channel_coeffs_count += 1
                        if abs(block_dct_coeffs[u, v]) < 0.01: # Threshold for "near-zero"
                            channel_near_zero_coeffs += 1
            total_coeffs_count += channel_coeffs_count
            near_zero_coeffs_count += channel_near_zero_coeffs

            if channel_coeffs_count > 0:
                channel_sparsity = (channel_near_zero_coeffs / channel_coeffs_count) * 100
                sparsity_metrics_per_channel.append(channel_sparsity)
                logging.info(f"Channel {c_idx} DCT Coefficient Sparsity (selected_uv range (1,6)x(1,6)): {channel_sparsity:.2f}% near-zero")
            else:
                logging.info(f"Channel {c_idx} DCT Coefficient Sparsity: No coefficients processed.")


        if total_coeffs_count > 0:
            avg_sparsity = (near_zero_coeffs_count / total_coeffs_count) * 100
            logging.info(f"Overall Average DCT Coefficient Sparsity (selected_uv range (1,6)x(1,6)): {avg_sparsity:.2f}% near-zero")
            if avg_sparsity > 50: # Threshold for warning
                logging.warning(f"High Overall DCT coefficient sparsity ({avg_sparsity:.2f}%) detected in selected_uv range; heavy JPEG compression may affect extraction.")
                actionable_error_message_parts.append(f"HighCoeffSparsityWarn={avg_sparsity:.2f}%")
            else:
                actionable_error_message_parts.append(f"CoeffSparsity={avg_sparsity:.2f}%")
        else:
            logging.warning("Could not calculate DCT coefficient sparsity (no coefficients found or processed).")
            actionable_error_message_parts.append("CoeffSparsity=Unavailable")
    else:
        logging.warning("Sparsity analysis skipped: raw_dct_blocks_per_channel is empty.")


    # Existing Averaging Logic (remains unchanged as per subtask)
    averaged_dct_blocks = []
    averaged_block_variances = []
    for block_idx in range(num_blocks_total):
        sum_dct_coeffs_block = np.zeros((8,8), dtype=np.float32)
        sum_variance_this_block = 0.0
        for c_idx in range(3): # Iterate through B, G, R channels
            sum_dct_coeffs_block += raw_dct_blocks_per_channel[c_idx][block_idx]
            sum_variance_this_block += block_variances_per_channel[c_idx][block_idx]

        # Equal averaging as per original logic and subtask requirement
        averaged_dct_blocks.append(sum_dct_coeffs_block / 3.0)
        averaged_block_variances.append(sum_variance_this_block / 3.0)

    logging.info(f"Averaged DCTs and variances computed for {len(averaged_dct_blocks)} blocks using equal channel contribution.")

    # Define coefficient pool using per-channel blocks
    selected_uv = [(u, v) for u in range(1, 6) for v in range(1, 6)]
    pool = [(bi, bj, c, u, v) for bi in range(num_blocks_i) # Add channel index c
            for bj in range(num_blocks_j) for c in range(3) for (u, v) in selected_uv]
    N = len(pool)

    if N == 0:
        err_msg = "Coefficient pool is empty. Cannot extract. Image may be too small or DCT selection failed."
        logging.error(err_msg)
        raise ValueError(err_msg) # Summary log handled by finally
    logging.info(f"Coefficient pool size (averaged channels): {N}")
    actionable_error_message_parts.append(f"PoolSize={N}")

    # --- Primary Length Extraction ---
    L_final = -1
    # Parameters for the primary length extraction loop
    current_K_len = initial_K_arg
    max_attempts_len = 21
    attempt_len = 0

    current_alpha_estimate_len = 1.0
    # Alpha parameters for length extraction (similar to message part's adaptive alpha)
    max_alpha_estimate_len = 5.0; min_alpha_estimate_len = 0.1; alpha_adjustment_factor_len = 1.5
    change_alpha_direction_threshold_len = 2; alpha_increases_done_len = 0; alpha_decreases_done_len = 0
    alpha_adjust_direction_len = 1
    last_L_primary_attempt = -1 # Store L from the last attempt of primary loop

    logging.info(f"Starting primary length extraction. Initial K={current_K_len}, Initial AlphaEst={current_alpha_estimate_len:.2f}, Max Attempts={max_attempts_len}")

    k_at_L_detection = current_K_len # Initialize for summary log
    alpha_at_L_detection = current_alpha_estimate_len # Initialize for summary log
    len_bits_correlations_for_detection = [] # To store correlations for the attempt that finds L

    while attempt_len < max_attempts_len:
        k_at_L_detection = current_K_len # Update K for this attempt
        alpha_at_L_detection = current_alpha_estimate_len # Update Alpha for this attempt
        current_attempt_len_bit_signals = [] # Store signals for this specific attempt

        len_encoded_extracted = []
        desc_len = f"Extracting length (attempt {attempt_len+1}/{max_attempts_len}, K={current_K_len}, alpha_est={current_alpha_estimate_len:.2f})"
        for i in tqdm(range(28), desc=desc_len, leave=False): # Length is 28 bits (Hamming of 16 bits)
            random.seed(key + i)
            idx_list = random.sample(range(N), min(current_K_len, N))
            p_val = [random.choice([1, -1]) for _ in range(len(idx_list))]
            sum_corr_numerator = 0.0
            sum_adaptive_denominator = 0.0 # Using the adaptive component logic for length too

            if not idx_list and current_K_len > 0: # Log if K > 0 but no indices (N might be < K)
                 logging.debug(f"Length Bit {i}: Empty idx_list with K={current_K_len}, N={N}")

            for k_loop_idx, pool_idx in enumerate(idx_list):
                bi_pool, bj_pool, c_pool, u, v = pool[pool_idx] # Now includes c_pool
                block_linear_idx = bi_pool * num_blocks_j + bj_pool

                # Use per-channel data and apply channel weights
                channel_key_map = ['B', 'G', 'R'] # B=0, G=1, R=2 due to cv2.split order
                weight_for_channel = channel_weights[channel_key_map[c_pool]]

                dct_coeff_val = raw_dct_blocks_per_channel[c_pool][block_linear_idx][u, v]
                variance_val = block_variances_per_channel[c_pool][block_linear_idx]

                adaptive_component = max((1 + variance_val / 1000.0), 0.01)

                sum_corr_numerator += dct_coeff_val * p_val[k_loop_idx] * weight_for_channel
                sum_adaptive_denominator += adaptive_component # Denominator is typically not weighted by channel strength directly

            avg_normalized_signal = sum_corr_numerator / sum_adaptive_denominator if sum_adaptive_denominator > 1e-9 else 0.0
            current_attempt_len_bit_signals.append(avg_normalized_signal) # Store for potential use
            bit = 1 if avg_normalized_signal > 0.0 else 0
            len_encoded_extracted.append(bit)

        # Decode length
        len_bits_extracted = []
        if len(len_encoded_extracted) == 28: # Ensure we have enough bits
            for j_idx in range(0, 28, 7):
                chunk = len_encoded_extracted[j_idx:j_idx+7]
                len_bits_extracted.extend(hamming_decode(chunk))
            L_decoded_from_attempt = int(''.join(map(str, len_bits_extracted[:16])), 2)
            last_L_primary_attempt = L_decoded_from_attempt # Store L from this attempt
            logging.info(f"Primary length attempt {attempt_len+1}: Decoded L_attempt={L_decoded_from_attempt}. K={current_K_len}, AlphaEst={current_alpha_estimate_len:.2f}")
        else:
            L_decoded_from_attempt = -1 # Indicate failure if not enough bits
            last_L_primary_attempt = -1
            logging.warning(f"Primary length attempt {attempt_len+1}: Failed to extract 28 bits for length. K={current_K_len}")

        # Validate L (must be multiple of 128 for AES blocks, between 256 and 2048 typical for message+IV+padding)
        is_L_valid = (L_decoded_from_attempt >= 256 and L_decoded_from_attempt <= 2048 and L_decoded_from_attempt % 128 == 0)
        if is_L_valid:
            L_final = L_decoded_from_attempt
            len_bits_correlations_for_detection = list(current_attempt_len_bit_signals) # Save signals from successful attempt
            logging.info(f"Primary length extraction successful on attempt {attempt_len+1}. Valid L={L_final} with K={current_K_len}, AlphaEst={current_alpha_estimate_len:.2f}.")
            break # Exit loop on success

        if attempt_len == max_attempts_len - 1 and not is_L_valid: # If this is the last attempt and L is still not valid
            len_bits_correlations_for_detection = list(current_attempt_len_bit_signals) # Save signals from the final attempt

        attempt_len += 1
        if attempt_len < max_attempts_len: # Only adjust if not the last attempt
            #logging.info(f"Primary length attempt {attempt_len} failed to yield compliant L. Current K={current_K_len}, AlphaEst={current_alpha_estimate_len:.2f}")
            # K/Alpha adjustment strategy (similar to message part)
            if attempt_len < max_attempts_len // 2 : # First half of attempts, focus on K
                increase_amount_len = max(1, initial_K_arg // 5) # Increase K by a fraction of initial K
                current_K_len = min(current_K_len + increase_amount_len, N) # Cap K at N
                logging.info(f"Adjusting params for next length attempt: new K={current_K_len} (AlphaEst={current_alpha_estimate_len:.2f})")
            else: # Second half, focus on Alpha, with small K nudges
                if alpha_adjust_direction_len == 1:
                    current_alpha_estimate_len = min(current_alpha_estimate_len * alpha_adjustment_factor_len, max_alpha_estimate_len)
                    alpha_increases_done_len += 1
                    if alpha_increases_done_len >= change_alpha_direction_threshold_len or current_alpha_estimate_len >= max_alpha_estimate_len:
                        alpha_adjust_direction_len = -1; alpha_decreases_done_len = 0
                else: # direction is -1
                    current_alpha_estimate_len = max(current_alpha_estimate_len / alpha_adjustment_factor_len, min_alpha_estimate_len)
                    alpha_decreases_done_len += 1
                    if alpha_decreases_done_len >= change_alpha_direction_threshold_len or current_alpha_estimate_len <= min_alpha_estimate_len:
                        alpha_adjust_direction_len = 1; alpha_increases_done_len = 0
                logging.info(f"Adjusting params for next length attempt: new alpha_est={current_alpha_estimate_len:.2f} (K={current_K_len})")

    final_L_value = L_final # Store L for summary log, even if -1
    actionable_error_message_parts.append(f"L_K={k_at_L_detection}")
    actionable_error_message_parts.append(f"L_Alpha={alpha_at_L_detection:.2f}")

    # --- Watermark Detection Logic (Primary Attempt) ---
    if L_final != -1 and len(len_bits_correlations_for_detection) == 28:
        avg_abs_correlation_len_bits = np.mean(np.abs(np.array(len_bits_correlations_for_detection)))
        detection_threshold_value = 0.075 * alpha_at_L_detection
        actionable_error_message_parts.append(f"AvgLenCorrPri={avg_abs_correlation_len_bits:.4f}")
        actionable_error_message_parts.append(f"DetectThreshPri={detection_threshold_value:.4f}")
        # Debug log for per-bit length correlations
        if len_bits_correlations_for_detection: # Ensure it's not empty before trying to format
            correlation_values_str = ", ".join([f"{val:.4f}" for val in len_bits_correlations_for_detection])
            logging.debug(f"Per-bit length correlations (L={L_final}, K={k_at_L_detection}, AlphaEst={alpha_at_L_detection:.2f}): [{correlation_values_str}]")
        logging.info(f"Watermark detection (Primary L): Avg Abs Correlation: {avg_abs_correlation_len_bits:.4f}, Threshold: {detection_threshold_value:.4f}, AlphaEst: {alpha_at_L_detection:.2f}, K: {k_at_L_detection}")
        if avg_abs_correlation_len_bits < detection_threshold_value:
            logging.error(f"Primary L={L_final} failed watermark detection. Avg correlation {avg_abs_correlation_len_bits:.4f} < threshold {detection_threshold_value:.4f}. Estimated alpha: {alpha_at_L_detection:.2f}, K for length: {k_at_L_detection}.")
            actionable_error_message_parts.append("DetectionPriFailed")
            # Raise ValueError immediately if primary L watermark detection fails and no fallback is planned or if this is a critical failure point.
            # However, current logic invalidates L_final and proceeds to fallback. If fallback also fails, a comprehensive error is raised there.
            # For clarity, we can raise a more specific error if this is the only point of failure for a *found* L.
            # The current structure will have L_final = -1 after this, leading to the generic "Could not determine L" if fallback also fails.
            # This is acceptable as the actionable_error_message_parts will contain "DetectionPriFailed".
            L_final = -1 # Invalidate L_final to trigger fallback or failure path
            final_L_value = -1 # Ensure summary reflects this invalidation
    elif L_final != -1:
        logging.warning("Watermark detection (Primary L) skipped: L_final found but length bit correlations not captured.")

    # --- Fallback Length Extraction ---
    max_confidence_from_failed_attempts = 0.0 # Tracks overall max confidence from any failed L attempt (primary or fallback)
    best_L_from_failed_attempts = -1
    k_at_best_failed_L = initial_K_arg
    alpha_at_best_failed_L = 1.0

    if len(len_bits_correlations_for_detection) == 28: # If primary attempts yielded correlations
         primary_last_attempt_confidence = np.mean(np.abs(np.array(len_bits_correlations_for_detection)))
         if primary_last_attempt_confidence > max_confidence_from_failed_attempts:
            max_confidence_from_failed_attempts = primary_last_attempt_confidence
            best_L_from_failed_attempts = last_L_primary_attempt if last_L_primary_attempt % 128 == 0 and 128 <= last_L_primary_attempt <= 2048 else -1
            k_at_best_failed_L = k_at_L_detection
            alpha_at_best_failed_L = alpha_at_L_detection

    if L_final == -1:
        logging.warning(f"Primary length extraction failed or L was invalidated by detection. Last primary L attempt: {last_L_primary_attempt}. Initiating fallback L search.")
        fallback_L_candidates = set()
        if last_L_primary_attempt > 0 :
            if 128 <= last_L_primary_attempt <= 2048 and last_L_primary_attempt % 128 == 0:
                 fallback_L_candidates.add(last_L_primary_attempt)
            base_L = (last_L_primary_attempt // 128) * 128
            if base_L > 0 and base_L <= 2048: fallback_L_candidates.add(base_L)
            if base_L + 128 <= 2048: fallback_L_candidates.add(base_L + 128)
            if base_L - 128 > 0: fallback_L_candidates.add(base_L-128)

        common_lengths = [256, 384, 512, 768, 1024] # Common AES block multiples for typical message sizes
        for cl in common_lengths:
            if 128 <= cl <= 2048: fallback_L_candidates.add(cl)

        # Filter and sort, limit count
        valid_fallback_candidates = sorted([l for l in list(fallback_L_candidates) if 128 <= l <= 2048 and l % 128 == 0])
        limited_fallback_candidates = valid_fallback_candidates[:7] # Limit to around 7
        actionable_error_message_parts.append(f"FallbackCands={limited_fallback_candidates}")

        if not limited_fallback_candidates:
            logging.warning("No suitable fallback L candidates generated.")
        else:
            logging.info(f"Generated fallback L candidates: {limited_fallback_candidates}")
            max_attempts_fallback = max(5, (max_attempts_len * 2) // 3) # e.g. 14 if max_attempts_len is 21

            for candidate_L in limited_fallback_candidates:
                k_for_fallback = k_at_L_detection # Use K from end of primary length attempts
                alpha_for_fallback = alpha_at_L_detection # Use Alpha from end of primary length attempts
                logging.info(f"Fallback: Attempting message extraction with L_candidate={candidate_L}, K_init={k_for_fallback}, Alpha_init={alpha_for_fallback:.2f}, MaxAttempts={max_attempts_fallback}")

                # Convert channel_weights dict to list [B, G, R] for helper function (consistent order)
                channel_weights_list_fb = [channel_weights['B'], channel_weights['G'], channel_weights['R']]
                fb_msg_data = _extract_message_for_length(
                    candidate_L, initial_K_arg, max_attempts_fallback, key, N, pool,
                    num_blocks_j, raw_dct_blocks_per_channel, block_variances_per_channel,
                    channel_weights_list_fb, alpha_for_fallback
                )
                fb_msg, fb_conf, fb_repl, fb_k, fb_alpha = fb_msg_data

                if fb_msg is not None:
                    L_final = candidate_L
                    final_L_value = L_final
                    final_extracted_message = fb_msg
                    final_message_confidence_score = fb_conf
                    final_num_replacements = fb_repl
                    k_at_msg_extraction = fb_k # K that worked for this fallback L
                    alpha_at_msg_extraction = fb_alpha # Alpha that worked
                    actionable_error_message_parts.append(f"FallbackSuccessL={L_final}")
                    logging.info(f"Fallback successful: Valid message extracted with L={L_final}. Confidence: {final_message_confidence_score*100:.1f}%. K={fb_k}, Alpha={fb_alpha:.2f}")
                    break # Exit fallback loop
                else: # Fallback for this L_candidate failed CRC or was all '?'
                    logging.warning(f"Fallback attempt with L_candidate={candidate_L} failed. Max confidence for this L: {fb_conf*100:.1f}%. K={fb_k}, Alpha={fb_alpha:.2f}")
                    if fb_conf > max_confidence_from_failed_attempts:
                        max_confidence_from_failed_attempts = fb_conf
                        best_L_from_failed_attempts = candidate_L
                        k_at_best_failed_L = fb_k
                        alpha_at_best_failed_L = fb_alpha

    # Final check if any L was successful (either primary or fallback)
    if L_final == -1:
        actionable_error_message_parts.append(f"LastLAttemptVal={last_L_primary_attempt}")
        if len(len_bits_correlations_for_detection) == 28: # From primary attempts
             avg_abs_correlation_last_primary = np.mean(np.abs(np.array(len_bits_correlations_for_detection)))
             actionable_error_message_parts.append(f"LastAvgLenCorrPri={avg_abs_correlation_last_primary:.4f}")

        # Include info about the best failing L if available
        if best_L_from_failed_attempts != -1:
            actionable_error_message_parts.append(f"BestFailedL={best_L_from_failed_attempts}")
            actionable_error_message_parts.append(f"BestFailedLConf={max_confidence_from_failed_attempts*100:.1f}%")
            actionable_error_message_parts.append(f"BestFailedLK={k_at_best_failed_L}")
            actionable_error_message_parts.append(f"BestFailedLAlpha={alpha_at_best_failed_L:.2f}")

        error_msg = (f"Extraction failed: Could not determine a valid message length after primary (Initial K={initial_K_arg}, Preprocessed={preprocess_actually_applied}) and fallback attempts. "
                     f"Max confidence for any failing L was {max_confidence_from_failed_attempts*100:.1f}% (for L={best_L_from_failed_attempts}, K={k_at_best_failed_L}, AlphaEst={alpha_at_best_failed_L:.2f}). "
                     f"Suggestions: Try with --preprocess if not used. Ensure the --key is correct. Verify image integrity and source. Consider different --K values. "
                     f"Details: {', '.join(actionable_error_message_parts)}.")
        logging.error(error_msg) # Log contains all details from actionable_error_message_parts
        # User-facing ValueError should be slightly more concise but guide to logs.
        user_error_msg = (f"Extraction failed: Could not determine a valid message length after primary and fallback attempts. Max signal confidence for any failing L was {max_confidence_from_failed_attempts*100:.1f}% (L={best_L_from_failed_attempts}). "
                          f"Key settings: Initial K={initial_K_arg}, Preprocessed={preprocess_actually_applied}. "
                          f"Suggestions: Try --preprocess if not used. Ensure --key is correct. Verify image. Consider different --K. Check logs for more details.")
        raise ValueError(user_error_msg)

    # If L_final is valid here, it means either primary L was good and passed detection, or a fallback L was successful.
    # The variables k_at_L_detection and alpha_at_L_detection are from the primary phase.
    # If fallback succeeded, k_at_msg_extraction and alpha_at_msg_extraction are set from the fallback's successful _extract_message_for_length call.
    # If primary succeeded, these will be set by the call to _extract_message_for_length below.

    if final_extracted_message is None: # If primary L was valid but message extraction hasn't happened yet (e.g. detection passed, now extract actual message)
        logging.info(f"Proceeding to message extraction with L={L_final}. Using K={initial_K_arg} (initial), AlphaEst={alpha_at_L_detection:.2f} (from L-phase) for message part.")
        # Convert channel_weights dict to list [B, G, R] for the main call to helper (if primary L was successful)
        channel_weights_list_main = [channel_weights['B'], channel_weights['G'], channel_weights['R']]
        extracted_msg_data = _extract_message_for_length(
            target_L=L_final,
            initial_K_for_msg=initial_K_arg,
            max_attempts_for_msg=max_attempts_len,
            key_param=key, N_pool_size=N, pool_details=pool, num_blocks_j_img=num_blocks_j,
            raw_dct_blocks_input=raw_dct_blocks_per_channel,         # Pass raw per-channel DCTs
            raw_block_variances_input=block_variances_per_channel,  # Pass raw per-channel variances
            active_channel_weights=channel_weights_list_main,        # Pass calculated channel weights
            passed_initial_alpha_estimate=alpha_at_L_detection
        )
        final_extracted_message, final_message_confidence_score, final_num_replacements, k_at_msg_extraction, alpha_at_msg_extraction = extracted_msg_data

    # Update actionable_error_message_parts with message extraction details
    actionable_error_message_parts.append(f"Msg_K={k_at_msg_extraction}") # This will be from primary L's msg extraction or fallback's successful one
    actionable_error_message_parts.append(f"Msg_Alpha={alpha_at_msg_extraction:.2f}")
    actionable_error_message_parts.append(f"Msg_Conf={final_message_confidence_score*100:.1f}%")

    if final_extracted_message is None:
        # This case implies L_final was found and passed watermark detection, but _extract_message_for_length failed for it.
        error_log_msg = (f"Extraction failed: Message could not be extracted for L={L_final} despite successful length detection and passing watermark signal check. "
                         f"Confidence for this L was {final_message_confidence_score*100:.1f}%. K used: {k_at_msg_extraction}, AlphaEst: {alpha_at_msg_extraction:.2f}. "
                         f"Details: {', '.join(actionable_error_message_parts)}.")
        logging.error(error_log_msg)
        user_error_msg = (f"Extraction failed for L={L_final}: Message could not be decoded despite valid length. "
                          f"Confidence: {final_message_confidence_score*100:.1f}%. "
                          f"Suggestions: Ensure correct --key. Image may be too noisy/compressed for message bits. Try --preprocess. Consider different --K. Check logs for details.")
        raise ValueError(user_error_msg)

    logging.info(f"Message extracted successfully: '{final_extracted_message}'. Confidence: {final_message_confidence_score*100:.1f}%. Replacements: {final_num_replacements}.")

    return final_extracted_message, final_message_confidence_score, final_num_replacements, final_L_value, k_at_L_detection, alpha_at_L_detection, k_at_msg_extraction, alpha_at_msg_extraction, preprocess_actually_applied, initial_K_arg

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
    extract_parser.add_argument("--blur-sigma", type=float, default=0.7, help="Sigma for Gaussian blur during pre-processing (default: 0.7)")
    extract_parser.add_argument("--clahe-clip-limit", type=float, default=1.5, help="Clip limit for CLAHE during pre-processing (default: 1.5)")
    extract_parser.add_argument("--variance-threshold", type=float, default=1000, help="Variance threshold to skip pre-processing (default: 1000)")

    args = parser.parse_args()

    # Variables for final summary log, initialized to sensible defaults
    summary_status = "Failed"
    summary_message = ""
    summary_confidence = 0.0
    summary_L = -1
    summary_K_L = args.K if args.command == 'extract' else 0 # K used for L detection
    summary_alpha_L = 1.0 # Alpha for L
    summary_K_msg = args.K if args.command == 'extract' else 0 # K for msg
    summary_alpha_msg = 1.0 # Alpha for msg
    summary_replacements = 0
    summary_preprocessed = False
    summary_initial_K = args.K if args.command == 'extract' else 0


    try:
        if args.command == "embed":
            if args.preprocess:
                logging.info("Pre-processing flag provided for embed, but no pre-processing is currently implemented for embedding.")
            embed(args.image, args.message, args.output, args.key, args.alpha, args.K, preprocess=False)
            print(f"Message embedded successfully into {args.output}")
            summary_status = "EmbedSuccess" # Special status for embed
            logging.info(f"Embedding Summary: Command=embed, Status={summary_status}, Output={args.output}, MessageLen={len(args.message)}, Key={args.key}, Alpha={args.alpha}, K={args.K}")

        elif args.command == "extract":
            summary_initial_K = args.K # Capture initial K for extract summary
            summary_preprocessed = args.preprocess # Will be updated if preprocess_actually_applied is different

            extracted_data = extract(args.image, args.key, args.K, preprocess=args.preprocess, blur_sigma=args.blur_sigma, clahe_clip_limit=args.clahe_clip_limit, variance_threshold=args.variance_threshold)

            # Unpack all returned values from the extract function
            (final_extracted_message, final_message_confidence_score, final_num_replacements,
            final_L_value, k_at_L_detection, alpha_at_L_detection, k_at_msg_extraction,
            alpha_at_msg_extraction, preprocess_actually_applied, initial_K_arg_from_func) = extracted_data

            summary_status = "Success"
            summary_message = final_extracted_message
            summary_confidence = final_message_confidence_score * 100
            summary_L = final_L_value
            summary_K_L = k_at_L_detection
            summary_alpha_L = alpha_at_L_detection
            summary_K_msg = k_at_msg_extraction
            summary_alpha_msg = alpha_at_msg_extraction
            summary_replacements = final_num_replacements
            summary_preprocessed = preprocess_actually_applied # Actual status

            print_msg = f"Extracted message: '{summary_message}', Confidence: {summary_confidence:.1f}%"
            if summary_replacements > 0:
                print_msg += f" (Note: {summary_replacements} non-ASCII char(s) replaced with '?')"
            print(print_msg)

    except FileNotFoundError as e:
        summary_status = "Failed_FileNotFound"
        logging.error(f"File not found: {str(e)}")
        print(f"Error - File not found: {str(e)}")
    except ValueError as e:
        summary_status = "Failed_ValueError"
        logging.error(f"Operation error: {str(e)}")
        print(f"Error: {str(e)}")
    except Exception as e:
        summary_status = "Failed_UnexpectedError"
        logging.error(f"An unexpected error occurred: {str(e)}", exc_info=True)
        print(f"An unexpected critical error occurred. Check logs for details: {str(e)}")
    finally:
        if args.command == "extract": # Only log detailed summary for extract
            final_summary_log = (
                f"Extraction Summary: Status={summary_status}, "
                f"Message='{summary_message}', Confidence={summary_confidence:.1f}%, "
                f"L={summary_L}, K_for_L={summary_K_L}, Alpha_for_L={summary_alpha_L:.2f}, "
                f"K_for_Msg={summary_K_msg}, Alpha_for_Msg={summary_alpha_msg:.2f}, "
                f"Replacements={summary_replacements}, Initial_K={summary_initial_K}, Pre-processed={summary_preprocessed}"
            )
            logging.info(final_summary_log)
            # Also print to console if failed, success message already printed
            if summary_status != "Success" and "Failed" in summary_status : # Check for "Failed" to avoid printing for EmbedSuccess
                 print(f"Final Status: {summary_status}. Check 'watermarking.log' for detailed execution summary.")