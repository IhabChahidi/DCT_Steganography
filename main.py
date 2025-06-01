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
from typing import List

# --- Constants ---
BLOCK_SIZE = 8
AES_KEY_SIZE_BYTES = 32
IV_SIZE_BYTES = 16
CRC_SIZE_BYTES = 4
HAMMING_DATA_BITS = 4
HAMMING_CODEWORD_BITS = 7
MESSAGE_LENGTH_BITS = 16  # For encoding the length of the message
ADAPTIVE_ALPHA_VARIANCE_DIVISOR = 1000.0  # For adaptive alpha calculation
MAX_MESSAGE_LENGTH_CHARS = 128

# Configure logging
logging.basicConfig(
    filename='watermarking.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def hamming_encode(data_bits: List[int]) -> List[int]:
    """
    Encodes 4 data bits into 7 bits using Hamming(7,4) code.

    Args:
        data_bits: A list of 4 integers representing the data bits (0 or 1).

    Returns:
        A list of 7 integers representing the Hamming encoded codeword.
    """
    assert len(data_bits) == HAMMING_DATA_BITS, f"Input must be {HAMMING_DATA_BITS} bits"
    d1, d2, d3, d4 = data_bits
    p1 = d1 ^ d2 ^ d4
    p2 = d1 ^ d3 ^ d4
    p3 = d2 ^ d3 ^ d4
    return [p1, p2, d1, p3, d2, d3, d4]

def hamming_decode(codeword: List[int]) -> List[int]:
    """
    Decodes a 7-bit Hamming codeword to 4 data bits, correcting single-bit errors.

    Args:
        codeword: A list of 7 integers representing the Hamming codeword.

    Returns:
        A list of 4 integers representing the decoded data bits.
        If a single-bit error is detected, it's corrected before decoding.
    """
    assert len(codeword) == HAMMING_CODEWORD_BITS, f"Input must be {HAMMING_CODEWORD_BITS} bits"
    p1, p2, d1, p3, d2, d3, d4 = codeword
    s1 = p1 ^ d1 ^ d2 ^ d4
    s2 = p2 ^ d1 ^ d3 ^ d4
    s3 = p3 ^ d2 ^ d3 ^ d4
    syndrome = s1 + 2 * s2 + 4 * s3
    if syndrome != 0 and syndrome <= 7:
        error_pos = syndrome - 1
        codeword[error_pos] ^= 1
    return [codeword[2], codeword[4], codeword[5], codeword[6]]

def generate_key_stream(length: int, key: int) -> List[int]:
    """Generate a pseudo-random key stream for encryption."""
    rand = Random(key)
    return [rand.randint(0, 255) for _ in range(length)]

def aes_encrypt(message_bytes: bytes, key: int) -> bytes:
    """
    Encrypts message bytes using AES-256-CBC encryption.

    A pseudo-random key stream is generated based on the provided integer key.
    An Initialization Vector (IV) is randomly generated and prepended to the
    encrypted output. PKCS7 padding is applied to the message before encryption.

    Args:
        message_bytes: The byte string to be encrypted.
        key: An integer key used to seed the pseudo-random key stream generator.

    Returns:
        A byte string containing the IV prepended to the encrypted message.
    """
    key_bytes = bytes(generate_key_stream(AES_KEY_SIZE_BYTES, key))  # AES-256
    iv = os.urandom(IV_SIZE_BYTES)  # IV
    cipher = Cipher(algorithms.AES(key_bytes), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    # Pad message to multiple of IV_SIZE_BYTES bytes
    pad_length = IV_SIZE_BYTES - (len(message_bytes) % IV_SIZE_BYTES)
    padded_message = message_bytes + bytes([pad_length] * pad_length)
    encrypted = encryptor.update(padded_message) + encryptor.finalize()
    logging.info(f"Message encrypted with AES-256, length: {len(encrypted)} bytes")
    return iv + encrypted  # Prepend IV for decryption

def aes_decrypt(encrypted_bytes: bytes, key: int) -> bytes:
    """
    Decrypts an AES-256-CBC encrypted byte string.

    The Initialization Vector (IV) is extracted from the beginning of the
    encrypted_bytes. The same integer key used for encryption must be provided
    to regenerate the key stream. PKCS7 padding is removed after decryption.

    Args:
        encrypted_bytes: The byte string to be decrypted (IV + ciphertext).
        key: An integer key used to seed the pseudo-random key stream generator.

    Returns:
        The original decrypted byte string.
    """
    key_bytes = bytes(generate_key_stream(AES_KEY_SIZE_BYTES, key))
    if len(encrypted_bytes) < IV_SIZE_BYTES:
        raise ValueError("Encrypted data is too short to contain an IV.")
    iv = encrypted_bytes[:IV_SIZE_BYTES]
    ciphertext = encrypted_bytes[IV_SIZE_BYTES:]

    if not ciphertext: # Check if ciphertext is empty after stripping IV
        raise ValueError("Ciphertext is empty after IV stripping. Cannot decrypt.")

    cipher = Cipher(algorithms.AES(key_bytes), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()

    try:
        padded_message = decryptor.update(ciphertext) + decryptor.finalize()
    except ValueError as e: # This can happen with bad padding or key
        logging.error(f"AES decryption failed during finalize: {str(e)}")
        raise ValueError("AES decryption failed, possibly due to incorrect key or corrupted data.") from e

    # Remove PKCS7 padding
    if not padded_message: # Should not be empty if finalize succeeded
        raise ValueError("Decryption resulted in empty message after finalize.")

    pad_length = padded_message[-1]
    if pad_length > IV_SIZE_BYTES or pad_length == 0: # Pad length must be between 1 and block size (16)
        raise ValueError("Invalid PKCS7 padding length.")
    if pad_length > len(padded_message):
        raise ValueError("Padding length exceeds message size.")

    # Check if all padding bytes are correct
    if not all(padded_message[-(pad_length):-1][i] == pad_length for i in range(pad_length -1)): # Check all but last byte
         if padded_message[-pad_length:] != bytes([pad_length] * pad_length):
            raise ValueError("Invalid PKCS7 padding bytes found.")

    message = padded_message[:-pad_length]
    logging.info(f"Message decrypted with AES-256, length: {len(message)} bytes")
    return message

def _load_and_prepare_image(image_path: str) -> tuple[np.ndarray, int, int]:
    """
    Loads, validates, and prepares an image for watermarking or extraction.

    Handles image format validation (PNG/JPEG), loading, color channel validation,
    and padding to ensure dimensions are multiples of BLOCK_SIZE.

    Args:
        image_path: Path to the input image.

    Returns:
        A tuple containing:
            - prepared_img: The loaded and padded image as a NumPy array.
            - h: Height of the padded image.
            - w: Width of the padded image.

    Raises:
        FileNotFoundError: If the image cannot be loaded.
        ValueError: If the image is not a 3-channel color image or not PNG/JPEG.
    """
    # Validate image format
    valid_extensions = ['.png', '.jpg', '.jpeg']
    if not any(image_path.lower().endswith(ext) for ext in valid_extensions):
        raise ValueError(f"Invalid image format: {image_path}. Only PNG and JPEG are supported.")

    # Check if file exists before trying to load
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found at path: {image_path}")

    # Load image
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        # This case means the file existed, but OpenCV could not read it as an image.
        raise ValueError(f"Could not read image at path: {image_path}. File may be corrupted or an unsupported format.")
    
    # Validate image properties
    if len(img.shape) != 3 or img.shape[2] != 3:
        raise ValueError(f"Input image at {image_path} must be a color image with 3 channels.")

    # Pad image to multiple of BLOCK_SIZE if necessary
    h_orig, w_orig, _ = img.shape
    pad_h = (BLOCK_SIZE - h_orig % BLOCK_SIZE) % BLOCK_SIZE
    pad_w = (BLOCK_SIZE - w_orig % BLOCK_SIZE) % BLOCK_SIZE
    if pad_h > 0 or pad_w > 0:
        img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)
    h, w, _ = img.shape
    logging.info(f"Image loaded: {image_path}, Original: {w_orig}x{h_orig}, Padded: {w}x{h}, Blocks: {h//BLOCK_SIZE}x{w//BLOCK_SIZE}")
    return img, h, w

def _prepare_message_for_embedding(message: str, key: int) -> List[int]:
    """
    Prepares the message for embedding by performing validation, encryption, and encoding.

    Steps:
    1. Validates message length.
    2. Encodes message to UTF-8 bytes.
    3. Computes CRC32 checksum.
    4. Appends CRC to message bytes.
    5. Encrypts the message_with_crc using AES.
    6. Converts encrypted bytes to a list of bits.
    7. Encodes the length of this bit list using Hamming code.
    8. Encodes the message bit list itself using Hamming code.
    9. Combines encoded length and encoded message bits.

    Args:
        message: The secret message string.
        key: Integer key for AES encryption.

    Returns:
        A list of integers representing the fully encoded bits ready for embedding.

    Raises:
        ValueError: If the message is too long.
    """
    # Validate message length
    if len(message) > MAX_MESSAGE_LENGTH_CHARS:
        raise ValueError(f"Message too long; maximum {MAX_MESSAGE_LENGTH_CHARS} characters")

    # Encrypt message with AES and compute CRC
    message_bytes = message.encode('utf-8')
    crc = compute_crc32(message_bytes)
    message_with_crc = message_bytes + crc.to_bytes(CRC_SIZE_BYTES, 'big')
    encrypted_bytes = aes_encrypt(message_with_crc, key)
    msg_bits = [int(b) for byte in encrypted_bytes for b in format(byte, '08b')]
    L = len(msg_bits)  # This now reflects the actual number of bits after padding
    logging.info(f"Message encrypted and converted to {L} bits")

    # Encode length (MESSAGE_LENGTH_BITS bits) with Hamming
    len_bits = [int(b) for b in bin(L)[2:].zfill(MESSAGE_LENGTH_BITS)]
    len_encoded = []
    for i in range(0, MESSAGE_LENGTH_BITS, HAMMING_DATA_BITS):
        chunk = len_bits[i:i+HAMMING_DATA_BITS]
        len_encoded.extend(hamming_encode(chunk))
    logging.info(f"Length {L} encoded to {len(len_encoded)} bits")

    # Encode message with Hamming
    num_chunks = (L + HAMMING_DATA_BITS - 1) // HAMMING_DATA_BITS
    msg_encoded = []
    for i in range(num_chunks):
        chunk = msg_bits[i*HAMMING_DATA_BITS:i*HAMMING_DATA_BITS+HAMMING_DATA_BITS]
        if len(chunk) < HAMMING_DATA_BITS:
            chunk.extend([0] * (HAMMING_DATA_BITS - len(chunk)))
        msg_encoded.extend(hamming_encode(chunk))
    logging.info(f"Message encoded to {len(msg_encoded)} bits")

    # Combine length and message bits
    encoded_bits = len_encoded + msg_encoded
    M = len(encoded_bits)
    logging.info(f"Total encoded bits for embedding: {M}")
    return encoded_bits

def _perform_dct_on_blocks(channels: List[np.ndarray], h: int, w: int) -> tuple[List[List[np.ndarray]], List[List[float]]]:
    """
    Performs DCT on 8x8 blocks for each channel of the image and computes block variances.

    Args:
        channels: A list of NumPy arrays, where each array represents a color channel of the image.
        h: Height of the image (must be divisible by BLOCK_SIZE).
        w: Width of the image (must be divisible by BLOCK_SIZE).

    Returns:
        A tuple containing:
            - dct_blocks: A list of lists of NumPy arrays. dct_blocks[c][i] is the
                          DCT result for the i-th block in the c-th channel.
            - block_variances: A list of lists of floats. block_variances[c][i] is the
                               variance of the i-th block in the c-th channel.
    """
    num_blocks_i, num_blocks_j = h // BLOCK_SIZE, w // BLOCK_SIZE
    dct_blocks = [[] for _ in range(len(channels))]
    block_variances = [[] for _ in range(len(channels))]

    for c, channel_data in enumerate(channels):
        for bi in range(num_blocks_i):
            for bj in range(num_blocks_j):
                block = channel_data[bi*BLOCK_SIZE:(bi+1)*BLOCK_SIZE, bj*BLOCK_SIZE:(bj+1)*BLOCK_SIZE]
                dct_blocks[c].append(cv2.dct(block.astype(np.float32)))
                block_variances[c].append(compute_block_variance(block))
    logging.info(f"DCT computed for {len(dct_blocks[0])} blocks per channel")
    return dct_blocks, block_variances

def _embed_bits_in_dct(
    dct_blocks: List[List[np.ndarray]],
    encoded_bits: List[int],
    key: int,
    alpha: float,
    K: int,
    pool: List[tuple[int, int, int, int, int]],
    block_variances: List[List[float]],
    num_blocks_j: int
) -> None:
    """
    Embeds the encoded bits into the DCT coefficients of the image blocks.

    Modifies dct_blocks in-place.

    Args:
        dct_blocks: DCT coefficients for each block and channel.
        encoded_bits: The list of bits (0s and 1s) to embed.
        key: Integer key for seeding random number generation.
        alpha: Initial embedding strength.
        K: Number of DCT coefficients to modify for each bit.
        pool: List of available (block_idx_i, block_idx_j, channel, u, v) for embedding.
        block_variances: Pre-computed variances of the image blocks.
        num_blocks_j: Number of blocks in the horizontal direction.
    """
    N = len(pool)
    for i in tqdm(range(len(encoded_bits)), desc="Embedding bits"):
        random.seed(key + i)
        idx_list = random.sample(range(N), min(K, N))
        p = [random.choice([1, -1]) for _ in range(len(idx_list))]
        m_i = 2 * encoded_bits[i] - 1  # Convert bit 0 to -1, 1 to 1

        for k_idx, pool_idx in enumerate(idx_list):
            bi, bj, c, u, v = pool[pool_idx]
            block_list_idx = bi * num_blocks_j + bj
            variance = block_variances[c][block_list_idx]
            adaptive_alpha = alpha * (1 + variance / ADAPTIVE_ALPHA_VARIANCE_DIVISOR)
            adaptive_alpha = min(adaptive_alpha, 1.0)  # Cap at 1.0

            dct_blocks[c][block_list_idx][u, v] += adaptive_alpha * m_i * p[k_idx]
        # Logging for each bit can be verbose, consider adjusting or removing if too noisy
        # logging.info(f"Bit {i} embedded with adaptive_alpha={adaptive_alpha:.4f}, K_used={len(idx_list)}")
    logging.info("All bits embedded into DCT coefficients.")

def _reconstruct_image_from_dct(
    dct_blocks: List[List[np.ndarray]],
    h: int,
    w: int,
    original_channels_shapes: List[tuple[int, ...]]
) -> np.ndarray:
    """
    Reconstructs the image from modified DCT blocks.

    Args:
        dct_blocks: List of lists of modified DCT coefficients.
        h: Height of the (padded) image.
        w: Width of the (padded) image.
        original_channels_shapes: List of shapes of the original image channels,
                                 used to create empty arrays of the correct size.

    Returns:
        The reconstructed watermarked image as a NumPy array.
    """
    num_blocks_i, num_blocks_j = h // BLOCK_SIZE, w // BLOCK_SIZE
    watermarked_channels = [np.zeros(shape, dtype=np.float32) for shape in original_channels_shapes]

    for c in range(len(dct_blocks)):
        for bi in range(num_blocks_i):
            for bj in range(num_blocks_j):
                block_list_idx = bi * num_blocks_j + bj
                idct_block = cv2.idct(dct_blocks[c][block_list_idx])
                watermarked_channels[c][bi*BLOCK_SIZE:(bi+1)*BLOCK_SIZE, bj*BLOCK_SIZE:(bj+1)*BLOCK_SIZE] = idct_block

    watermarked_img = cv2.merge([np.clip(channel, 0, 255).astype(np.uint8) for channel in watermarked_channels])
    # The actual saving is done in the main embed function, so just return the image
    return watermarked_img

def _extract_bits_from_dct(
    dct_blocks: List[List[np.ndarray]],
    key: int,
    initial_K: int,
    pool: List[tuple[int, int, int, int, int]],
    num_blocks_j: int,
    len_encoded_bits_to_extract: int
) -> tuple[List[int], List[int], int]:
    """
    Extracts encoded length and message bits from DCT coefficients.

    Includes adaptive K for extraction attempts for the length part.
    The message part is extracted once a valid length L is determined.

    Args:
        dct_blocks: DCT coefficients for each block and channel.
        key: Integer key for seeding random number generation.
        initial_K: Initial number of DCT coefficients to check per bit for length extraction.
        pool: List of available (block_idx_i, block_idx_j, channel, u, v) for extraction.
        num_blocks_j: Number of blocks in the horizontal direction.
        len_encoded_bits_to_extract: The number of encoded bits to extract for the length.

    Returns:
        A tuple containing:
            - len_encoded_extracted: The extracted list of encoded length bits.
            - msg_encoded_extracted: The extracted list of encoded message bits.
            - L: The decoded message length in bits (after Hamming decoding of length part).

    Raises:
        ValueError: If a valid length cannot be extracted after maximum attempts.
    """
    N = len(pool)
    max_attempts_len = 5
    attempt_len = 0
    current_K_len = initial_K
    len_encoded_extracted = []
    L = 0 # Decoded length
    msg_encoded_extracted = []

    logging.info(f"Starting extraction of {len_encoded_bits_to_extract} encoded length bits.")
    while attempt_len < max_attempts_len:
        len_encoded_extracted.clear()
        logging.info(f"Length extraction attempt {attempt_len + 1}/{max_attempts_len} with K={current_K_len}.")
        for i in tqdm(range(len_encoded_bits_to_extract), desc=f"Extracting length bits (K={current_K_len})"):
            random.seed(key + i) # Key offset for length bits is 0
            idx_list = random.sample(range(N), min(current_K_len, N))
            p = [random.choice([1, -1]) for _ in range(len(idx_list))]
            sum_corr = 0
            count = 0
            for k_idx, pool_idx in enumerate(idx_list):
                bi, bj, c_idx, u, v = pool[pool_idx] # c_idx is channel index
                block_list_idx = bi * num_blocks_j + bj
                sum_corr += dct_blocks[c_idx][block_list_idx][u, v] * p[k_idx]
                count += 1

            extracted_bit = 1 if sum_corr > 0 else 0
            if count == 0:
                 extracted_bit = 0
                 logging.warning(f"Count was 0 during length bit {i} extraction (K={current_K_len}, N={N}).")
            len_encoded_extracted.append(extracted_bit)

        # Decode length
        len_bits_decoded = []
        if len(len_encoded_extracted) < len_encoded_bits_to_extract:
            logging.warning("Length extraction yielded too few bits before decoding.") # Should not happen
            attempt_len += 1
            current_K_len = min(current_K_len * 2, N)
            continue

        for j_idx in range(0, len_encoded_bits_to_extract, HAMMING_CODEWORD_BITS):
            chunk = len_encoded_extracted[j_idx : j_idx + HAMMING_CODEWORD_BITS]
            if len(chunk) < HAMMING_CODEWORD_BITS: # Should not happen if len_encoded_bits_to_extract is multiple of HAMMING_CODEWORD_BITS
                 logging.warning(f"Skipping incomplete chunk of {len(chunk)} for length decoding.")
                 break
            len_bits_decoded.extend(hamming_decode(chunk))

        if len(len_bits_decoded) < MESSAGE_LENGTH_BITS:
            logging.warning(f"Decoded length bit stream too short: {len(len_bits_decoded)} bits, expected {MESSAGE_LENGTH_BITS}.")
            attempt_len += 1
            current_K_len = min(current_K_len * 2, N)
            continue

        L = int(''.join(map(str, len_bits_decoded[:MESSAGE_LENGTH_BITS])), 2)
        logging.info(f"Attempt {attempt_len + 1}: Extracted and decoded L = {L} bits.")

        max_possible_L_bits = (MAX_MESSAGE_LENGTH_CHARS * 4 + CRC_SIZE_BYTES + IV_SIZE_BYTES) * 8
        if L > 0 and L <= max_possible_L_bits and L % (IV_SIZE_BYTES * 8) == 0:
            logging.info(f"Valid L={L} found. Proceeding to extract message bits.")

            # Calculate how many encoded message bits to extract
            num_msg_data_chunks = (L + HAMMING_DATA_BITS - 1) // HAMMING_DATA_BITS
            msg_encoded_bits_to_extract = num_msg_data_chunks * HAMMING_CODEWORD_BITS
            logging.info(f"Expecting {msg_encoded_bits_to_extract} encoded message bits for L={L}.")

            msg_encoded_extracted.clear()
            # Start extracting message bits from key + len_encoded_bits_to_extract
            # For message bits, we typically use the initial_K and don't iterate K unless full decoding fails later.
            current_K_msg = initial_K # Or a different K strategy if needed
            for i_msg in tqdm(range(msg_encoded_bits_to_extract), desc=f"Extracting message bits (L={L}, K={current_K_msg})"):
                actual_key_idx = key + len_encoded_bits_to_extract + i_msg # Key offset for message bits
                random.seed(actual_key_idx)
                idx_list_msg = random.sample(range(N), min(current_K_msg, N))
                p_msg = [random.choice([1, -1]) for _ in range(len(idx_list_msg))]
                sum_corr_msg = 0
                count_msg = 0
                for k_msg_idx, pool_idx_msg in enumerate(idx_list_msg):
                    bi_msg, bj_msg, c_idx_msg, u_msg, v_msg = pool[pool_idx_msg]
                    block_list_idx_msg = bi_msg * num_blocks_j + bj_msg
                    sum_corr_msg += dct_blocks[c_idx_msg][block_list_idx_msg][u_msg, v_msg] * p_msg[k_msg_idx]
                    count_msg += 1

                extracted_bit_msg = 1 if sum_corr_msg > 0 else 0
                if count_msg == 0:
                     extracted_bit_msg = 0
                     logging.warning(f"Count was 0 during message bit {i_msg} extraction (K={current_K_msg}, N={N}).")
                msg_encoded_extracted.append(extracted_bit_msg)

            if len(msg_encoded_extracted) == msg_encoded_bits_to_extract:
                logging.info(f"Successfully extracted {len(msg_encoded_extracted)} encoded message bits.")
                return len_encoded_extracted, msg_encoded_extracted, L # Success
            else:
                logging.warning(f"Extracted {len(msg_encoded_extracted)} message bits, but expected {msg_encoded_bits_to_extract}. Retrying length extraction.")
                # This case implies an issue, so we loop to retry length extraction with a new K_len.

        attempt_len += 1
        current_K_len = min(current_K_len * 2, N)
        if attempt_len < max_attempts_len:
            logging.warning(f"Invalid L ({L}) or message bit extraction failed. Adjusting K_len to {current_K_len} for next length attempt.")
        else:
            logging.error(f"Failed to extract valid L after {max_attempts_len} attempts.")

    raise ValueError("Failed to extract valid message length L after maximum attempts.")

def _decode_extracted_message(
    msg_encoded_bits: List[int], # These are the Hamming encoded message bits
    L: int, # This is the actual length of the original message bits (after AES, before Hamming)
    key: int
) -> str:
    """
    Decodes the extracted message bits, decrypts, and verifies CRC.

    Args:
        msg_encoded_bits: List of Hamming encoded message bits.
        L: The actual length in bits of the message payload (after AES encryption, before Hamming).
        key: Integer key for AES decryption.

    Returns:
        The extracted and verified secret message string.

    Raises:
        ValueError: If decoding, decryption, or CRC verification fails.
    """
    logging.info(f"Starting decoding of {len(msg_encoded_bits)} encoded message bits for L={L}.")
    msg_bits_extracted = []
    for j in range(0, len(msg_encoded_bits), HAMMING_CODEWORD_BITS):
        chunk = msg_encoded_bits[j : j + HAMMING_CODEWORD_BITS]
        if len(chunk) < HAMMING_CODEWORD_BITS:
            # This might happen if msg_encoded_bits is not a perfect multiple,
            # though it should be based on L.
            logging.warning(f"Skipping incomplete chunk of {len(chunk)} bits during message decoding.")
            break
        msg_bits_extracted.extend(hamming_decode(chunk))
    
    if len(msg_bits_extracted) < L:
        raise ValueError(f"Not enough message bits after Hamming decoding ({len(msg_bits_extracted)}) for expected length L={L}")
    
    # Truncate to actual length L
    msg_bits_extracted = msg_bits_extracted[:L]
    logging.info(f"Successfully Hamming-decoded {L} message bits.")

    if L % 8 != 0:
        # This should have been caught by L validation in _extract_bits_from_dct
        raise ValueError(f"Decoded message length L={L} is not a multiple of 8. Cannot convert to bytes.")

    # Convert bits to bytes
    encrypted_payload_bytes_list = []
    for i in range(0, L, 8):
        byte_str = ''.join(map(str, msg_bits_extracted[i : i + 8]))
        encrypted_payload_bytes_list.append(int(byte_str, 2))
    encrypted_payload_bytes = bytes(encrypted_payload_bytes_list)
    logging.info(f"Converted message bits to {len(encrypted_payload_bytes)} encrypted bytes.")

    # Decrypt (AES handles IV stripping and unpadding)
    try:
        decrypted_payload_bytes = aes_decrypt(encrypted_payload_bytes, key)
    except Exception as e: # aes_decrypt might raise various crypto errors
        logging.error(f"AES decryption failed: {str(e)}")
        raise ValueError(f"AES decryption failed: {str(e)}")
    logging.info(f"AES decryption resulted in {len(decrypted_payload_bytes)} bytes of payload (message + CRC).")

    # Verify CRC
    if len(decrypted_payload_bytes) < CRC_SIZE_BYTES:
        raise ValueError(f"Decrypted payload is too short ({len(decrypted_payload_bytes)} bytes) to contain CRC ({CRC_SIZE_BYTES} bytes).")

    received_crc_bytes = decrypted_payload_bytes[-CRC_SIZE_BYTES:]
    actual_message_bytes = decrypted_payload_bytes[:-CRC_SIZE_BYTES]
    received_crc = int.from_bytes(received_crc_bytes, 'big')
    computed_crc = compute_crc32(actual_message_bytes)

    if received_crc != computed_crc:
        logging.error(f"CRC mismatch: received {received_crc}, computed {computed_crc}.") # Log as error
        raise ValueError("CRC mismatch, data integrity compromised.")
    logging.info("CRC verified successfully.")

    # Decode message from bytes to string
    try:
        extracted_message_str = actual_message_bytes.decode('utf-8')
        # A more robust check for printability might be desired depending on expected content
        # Allow common whitespace characters like space, tab, newline, carriage return.
        allowed_whitespace = " \t\n\r"
        if not all(32 <= ord(char) <= 126 or char in allowed_whitespace for char in extracted_message_str if char):
            logging.warning(f"Extracted message contains potentially non-printable or unusual characters: '{extracted_message_str[:100]}...'")
    except UnicodeDecodeError as e:
        logging.error(f"UTF-8 decoding error for message: '{actual_message_bytes[:100]}...': {str(e)}")
        raise ValueError("Failed to decode message (UTF-8 decoding error).") from e

    logging.info(f"Successfully decoded message: '{extracted_message_str}'")
    return extracted_message_str

def compute_crc32(data: bytes) -> int:
    """Compute CRC-32 checksum for data."""
    return zlib.crc32(data) & 0xFFFFFFFF

def compute_block_variance(block: np.ndarray) -> float:
    """Compute variance of an 8x8 block for adaptive embedding."""
    return np.var(block)

def embed(image_path: str, message: str, output_path: str, key: int, alpha: float, K: int) -> None:
    """
    Embeds a secret message into a color image using DCT-based spread spectrum watermarking.

    The process involves:
    1.  Validating image format (PNG/JPEG) and message length.
    2.  Padding the image to be divisible by block size.
    3.  Splitting the image into color channels (B, G, R).
    4.  Dividing each channel into 8x8 blocks and applying DCT.
    5.  Calculating block variances for adaptive embedding strength.
    6.  Encrypting the message (with CRC) using AES-256-CBC.
    7.  Encoding the length of the encrypted message and the message itself using Hamming codes.
    8.  Embedding the encoded bits into selected DCT coefficients of image blocks.
        The embedding strength (`alpha`) is adapted based on block variance.
        `K` coefficients are used to embed each bit for robustness.
    9.  Reconstructing the image from modified DCT blocks and saving the watermarked image.

    Args:
        image_path: Path to the input color image (PNG or JPEG).
        message: The secret message string to embed (max 128 characters).
        output_path: Path to save the watermarked output image.
        key: An integer key for pseudo-random sequence generation (for DCT coefficient selection)
             and AES encryption.
        alpha: Initial embedding strength (float, typically between 0.01 and 1.0).
               Controls the magnitude of modification to DCT coefficients.
        K: The number of DCT coefficients to use for embedding each bit of the message.
           Higher K increases robustness but may also increase visual distortion.

    Raises:
        FileNotFoundError: If the input image cannot be loaded.
        ValueError: If the image is not a 3-channel color image, if the message is too long,
                    or if the image format is not PNG/JPEG.
    """
    img, h, w = _load_and_prepare_image(image_path)
    encoded_bits = _prepare_message_for_embedding(message, key)

    # Split into channels
    b, g, r = cv2.split(img)
    dct_blocks, block_variances = _perform_dct_on_blocks([b, g, r], h, w)

    # Select mid-frequency coefficients
    num_blocks_i, num_blocks_j = h // BLOCK_SIZE, w // BLOCK_SIZE # Recalculate or pass from _perform_dct_on_blocks
    selected_uv = [(u, v) for u in range(1, BLOCK_SIZE) for v in range(1, BLOCK_SIZE)]
    pool = [(bi, bj, c, u, v) for bi in range(num_blocks_i)
            for bj in range(num_blocks_j) for c in range(3) for (u, v) in selected_uv]
    N = len(pool)
    logging.info(f"Coefficient pool size: {N}")

    # Embed each bit with adaptive alpha
    _embed_bits_in_dct(dct_blocks, encoded_bits, key, alpha, K, pool, block_variances, num_blocks_j)

    # Reconstruct image
    # Pass original channel shapes for creating watermarked_channels
    watermarked_img = _reconstruct_image_from_dct(dct_blocks, h, w, [ch.shape for ch in [b, g, r]])
    cv2.imwrite(output_path, watermarked_img)
    logging.info(f"Image saved to {output_path}")

def extract(image_path: str, key: int, K: int) -> str:
    """
    Extracts a secret message from a DCT-based spread spectrum watermarked color image.

    The process involves:
    1.  Validating image format (PNG/JPEG).
    2.  Padding the image if necessary.
    3.  Splitting the image into color channels and applying DCT to 8x8 blocks.
    4.  Extracting the encoded length of the message by correlating with DCT coefficients
        selected using the provided key. This step involves an adaptive process to find
        the optimal alpha and K values if initial extraction fails.
    5.  Decoding the length using Hamming codes.
    6.  Extracting the encoded message bits similarly.
    7.  Decoding the message bits using Hamming codes.
    8.  Converting the bits to bytes and decrypting using AES-256-CBC.
    9.  Verifying the CRC32 checksum of the decrypted message.
    10. Returning the decoded message string.

    Args:
        image_path: Path to the watermarked color image (PNG or JPEG).
        key: The integer key used during embedding for pseudo-random sequence generation
             and AES decryption.
        K: The initial number of DCT coefficients per bit to check during extraction.
           The extraction process might adapt this value.

    Returns:
        The extracted secret message string.

    Raises:
        FileNotFoundError: If the input image cannot be loaded.
        ValueError: If the image is not a 3-channel color image, if the image format is
                    not PNG/JPEG, or if message extraction fails after multiple attempts
                    (e.g., due to incorrect key, severe image degradation, or no message).
    """
    img, h, w = _load_and_prepare_image(image_path)

    # Split into channels
    b, g, r = cv2.split(img)
    # Reuse _perform_dct_on_blocks, ignore block_variances for now as extract doesn't use adaptive alpha based on it
    dct_blocks, _ = _perform_dct_on_blocks([b, g, r], h, w) # _ are block_variances

    # Define coefficient pool
    num_blocks_i, num_blocks_j = h // BLOCK_SIZE, w // BLOCK_SIZE
    selected_uv = [(u, v) for u in range(1, BLOCK_SIZE) for v in range(1, BLOCK_SIZE)]
    pool = [(bi, bj, c_idx, u, v) for bi in range(num_blocks_i) # c_idx instead of c
            for bj in range(num_blocks_j) for c_idx in range(3) for (u, v) in selected_uv]
    N = len(pool)
    logging.info(f"Coefficient pool size: {N}")

    len_encoded_bits_to_extract = MESSAGE_LENGTH_BITS * HAMMING_CODEWORD_BITS // HAMMING_DATA_BITS

    # Call the new extraction function
    # K here is initial_K for length extraction
    _, msg_encoded_extracted, L = _extract_bits_from_dct(
        dct_blocks, key, K, pool, num_blocks_j, len_encoded_bits_to_extract
    )

    # The old extraction loop for length and message is now replaced by the call above.
    # The adaptive K for message part and retry logic for full decode will be in _decode_extracted_message

    # Call the new decoding function
    extracted_message_str = _decode_extracted_message(msg_encoded_extracted, L, key)
    return extracted_message_str
    current_alpha = 0.1
    current_K = K
    max_attempts = 5
    attempt = 0

    while attempt < max_attempts:
        len_encoded_extracted = []
        for i in tqdm(range(len_encoded_bits_to_extract), desc=f"Extracting length (attempt {attempt+1})"):
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
        for j in range(0, len_encoded_bits_to_extract, HAMMING_CODEWORD_BITS):
            chunk = len_encoded_extracted[j:j+HAMMING_CODEWORD_BITS]
            len_bits_extracted.extend(hamming_decode(chunk))
        L = int(''.join(map(str, len_bits_extracted[:MESSAGE_LENGTH_BITS])), 2)
        logging.info(f"Extracted length (decoded): {L}")

        # Validate length and ensure it's a multiple of IV_SIZE_BYTES * 8 bits (IV_SIZE_BYTES bytes)
        if L > 0 and L <= (MAX_MESSAGE_LENGTH_CHARS * 8 * HAMMING_CODEWORD_BITS // HAMMING_DATA_BITS) and L % (IV_SIZE_BYTES * 8) == 0:  # Ensure L corresponds to a multiple of IV_SIZE_BYTES bytes
            break
        attempt += 1
        current_alpha = min(current_alpha * 2, max_alpha)
        current_K = min(current_K * 2, N)
        logging.info(f"Attempt {attempt} failed, adjusting alpha to {current_alpha}, K to {current_K}")

    if attempt >= max_attempts:
        raise ValueError("Failed to extract valid length after maximum attempts")

    # Extract message bits
    num_chunks = (L + HAMMING_DATA_BITS - 1) // HAMMING_DATA_BITS
    M_msg = num_chunks * HAMMING_CODEWORD_BITS
    msg_encoded_extracted = []
    attempt = 0
    current_alpha = 0.1
    current_K = K

    while attempt < max_attempts:
        msg_encoded_extracted = []
        for i in tqdm(range(len_encoded_bits_to_extract, len_encoded_bits_to_extract + M_msg), desc=f"Extracting message (attempt {attempt+1})"):
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
            logging.info(f"Message bit {i-len_encoded_bits_to_extract}, sum_corr={sum_corr:.4f}, c={c:.4f}, bit={bit}, alpha={current_alpha}, K={current_K}")

        # Decode message
        msg_bits_extracted = []
        for j in range(0, len(msg_encoded_extracted), HAMMING_CODEWORD_BITS):
            chunk = msg_encoded_extracted[j:j+HAMMING_CODEWORD_BITS]
            if len(chunk) < HAMMING_CODEWORD_BITS:
                break
            msg_bits_extracted.extend(hamming_decode(chunk))
        msg_bits_extracted = msg_bits_extracted[:L]

        # Convert to bytes and decrypt
        extracted_bytes = [int(''.join(map(str, msg_bits_extracted[i:i+8])), 2) for i in range(0, L, 8)] # Assuming 8 bits per byte
        decrypted_bytes = aes_decrypt(bytes(extracted_bytes), key)

        # Verify CRC and decode
        try:
            received_crc = int.from_bytes(decrypted_bytes[-CRC_SIZE_BYTES:], 'big')
            message_bytes = decrypted_bytes[:-CRC_SIZE_BYTES]
            computed_crc = compute_crc32(message_bytes)
            if received_crc == computed_crc:
                msg_extracted = message_bytes.decode('utf-8')
                if all(32 <= ord(char) <= 126 for char in msg_extracted): # Check for printable ASCII
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
    embed_parser.add_argument("--alpha", type=float, default=1.0, help="Initial embedding strength (default: 1.0)")
    embed_parser.add_argument("--K", type=int, default=500, help="Initial coefficients per bit (default: 500)")

    # Extract command
    extract_parser = subparsers.add_parser("extract", help="Extract a message from a color image")
    extract_parser.add_argument("--image", required=True, help="Input watermarked color image path (PNG/JPEG)")
    extract_parser.add_argument("--key", type=int, required=True, help="Key used during embedding")
    extract_parser.add_argument("--K", type=int, default=500, help="Initial coefficients per bit (default: 500)")

    args = parser.parse_args()

    # Validate command-line arguments
    if args.command == "embed":
        if not (0.0 < args.alpha):
            parser.error("Embedding strength alpha must be positive.") # parser.error exits
        if args.alpha > 5.0: # A more reasonable upper bound than 10.0 for typical images
            logging.warning(f"High embedding strength alpha={args.alpha} may cause significant image distortion.")
        if args.K <= 0:
            parser.error("Number of coefficients K must be a positive integer.")
    elif args.command == "extract":
        if args.K <= 0:
            parser.error("Number of coefficients K must be a positive integer.")

    try:
        if args.command == "embed":
            embed(args.image, args.message, args.output, args.key, args.alpha, args.K)
            print(f"Message embedded successfully into {args.output}")
        elif args.command == "extract":
            message = extract(args.image, args.key, args.K)
            print(f"Extracted message: {message}")
    except Exception as e:
        logging.error(f"Operation failed: {str(e)}") # More generic message for logging
        print(f"Error: {str(e)}") # Print specific error to console