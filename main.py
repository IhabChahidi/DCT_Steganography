import cv2
import numpy as np
import random
import argparse
import logging
from random import Random
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
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

def encode_data_with_hamming(data_bits: list[int]) -> list[int]:
    """Encodes a list of data bits using Hamming(7,4) code with padding."""
    encoded_bits_all = []
    num_total_bits = len(data_bits)
    for i in range(0, num_total_bits, 4):
        chunk = data_bits[i:i+4]
        if len(chunk) < 4:
            chunk.extend([0] * (4 - len(chunk)))  # Pad with zeros
        encoded_bits_all.extend(hamming_encode(chunk))
    return encoded_bits_all

def decode_data_with_hamming(encoded_bits: list[int]) -> list[int]:
    """Decodes a list of Hamming-encoded bits."""
    if len(encoded_bits) % 7 != 0:
        # Or handle this more gracefully depending on requirements, e.g., log and truncate
        raise ValueError("Length of encoded_bits must be a multiple of 7.")

    decoded_bits_all = []
    num_total_encoded_bits = len(encoded_bits)
    for i in range(0, num_total_encoded_bits, 7):
        chunk = encoded_bits[i:i+7]
        decoded_bits_all.extend(hamming_decode(chunk))
    return decoded_bits_all

def derive_key(password: str, salt: bytes, length: int) -> bytes:
    """Derive a key using PBKDF2."""
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=length,
        salt=salt,
        iterations=100000, # NIST recommended minimum
        backend=default_backend()
    )
    return kdf.derive(password.encode('utf-8'))

def aes_encrypt(message_bytes, password: str):
    """Encrypt the message bytes using AES-256-CBC and PBKDF2."""
    salt = os.urandom(16)  # Generate a new salt for each encryption
    key_bytes = derive_key(password, salt, 32)  # 32 bytes for AES-256
    iv = os.urandom(16)  # 16-byte IV
    cipher = Cipher(algorithms.AES(key_bytes), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    # Pad message to multiple of 16 bytes
    pad_length = 16 - (len(message_bytes) % 16)
    padded_message = message_bytes + bytes([pad_length] * pad_length)
    encrypted = encryptor.update(padded_message) + encryptor.finalize()
    logging.info(f"Message encrypted with AES-256, length: {len(encrypted)} bytes (excluding salt and IV)")
    return salt + iv + encrypted

def aes_decrypt(encrypted_bytes_with_salt_iv, password: str):
    """Decrypt the encrypted bytes using AES-256-CBC and PBKDF2."""
    salt = encrypted_bytes_with_salt_iv[:16]
    iv = encrypted_bytes_with_salt_iv[16:32]
    ciphertext = encrypted_bytes_with_salt_iv[32:]

    key_bytes = derive_key(password, salt, 32)
    cipher = Cipher(algorithms.AES(key_bytes), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    padded_message = decryptor.update(ciphertext) + decryptor.finalize()
    # Remove padding
    pad_length = padded_message[-1]
    if pad_length > 16 or pad_length == 0: # Basic check for padding validity
        # Ensure this part of the message is not empty before accessing its last byte
        if not padded_message:
            raise ValueError("Padded message is empty, cannot determine padding length.")
        # Check if pad_length is feasible
        if pad_length > len(padded_message):
            raise ValueError("Invalid padding: pad_length is greater than message length.")
        raise ValueError("Invalid padding length during decryption.")
    message = padded_message[:-pad_length]
    logging.info(f"Message decrypted with AES-256, length: {len(message)} bytes")
    return message

def compute_crc32(data):
    """Compute CRC-32 checksum for data."""
    return zlib.crc32(data) & 0xFFFFFFFF

def compute_block_variance(block):
    """Compute variance of an 8x8 block for adaptive embedding."""
    return np.var(block)

def apply_dct_to_image_channels(image_channels: list[np.ndarray]) -> list[list[np.ndarray]]:
    """
    Applies DCT to each 8x8 block of each image channel.
    Assumes image_channels are already padded to be multiples of 8.
    Returns a list of lists, where each inner list contains flat DCT blocks for a channel.
    """
    all_channels_dct_blocks = []
    for channel in image_channels:
        h, w = channel.shape
        num_blocks_i, num_blocks_j = h // 8, w // 8
        channel_dct_blocks = []
        for bi in range(num_blocks_i):
            for bj in range(num_blocks_j):
                block = channel[bi*8:(bi+1)*8, bj*8:(bj+1)*8]
                channel_dct_blocks.append(cv2.dct(block.astype(np.float32)))
        all_channels_dct_blocks.append(channel_dct_blocks)
    logging.info(f"DCT applied to {len(all_channels_dct_blocks[0]) if all_channels_dct_blocks else 0} blocks per channel for {len(all_channels_dct_blocks)} channels.")
    return all_channels_dct_blocks

def apply_idct_to_dct_blocks(dct_channel_blocks: list[list[np.ndarray]], padded_h: int, padded_w: int) -> list[np.ndarray]:
    """
    Applies IDCT to DCT blocks and reconstructs image channels.
    dct_channel_blocks: list of lists of DCT blocks (flat list per channel).
    padded_h, padded_w: dimensions of the padded image.
    Returns a list of reconstructed image channels (clipped to 0-255, np.uint8).
    """
    reconstructed_channels = []
    num_blocks_i, num_blocks_j = padded_h // 8, padded_w // 8

    for channel_idx, flat_dct_blocks in enumerate(dct_channel_blocks):
        if len(flat_dct_blocks) != num_blocks_i * num_blocks_j:
            raise ValueError(f"Channel {channel_idx}: Number of DCT blocks ({len(flat_dct_blocks)}) does not match expected ({num_blocks_i * num_blocks_j}).")

        reconstructed_channel = np.zeros((padded_h, padded_w), dtype=np.float32)
        block_idx = 0
        for bi in range(num_blocks_i):
            for bj in range(num_blocks_j):
                idct_block = cv2.idct(flat_dct_blocks[block_idx])
                reconstructed_channel[bi*8:(bi+1)*8, bj*8:(bj+1)*8] = idct_block
                block_idx += 1
        reconstructed_channels.append(np.clip(reconstructed_channel, 0, 255).astype(np.uint8))
    logging.info(f"IDCT applied and channels reconstructed for {len(reconstructed_channels)} channels.")
    return reconstructed_channels

def prepare_message_for_embedding(message_str: str, password: str) -> tuple[list[int], int]:
    """
    Prepares the message for embedding: CRC -> Encrypt -> Bits -> Hamming Encode Length -> Hamming Encode Payload.
    Returns the final list of bits to embed and the length of the payload (salt+iv+ciphertext) in bits.
    """
    message_bytes = message_str.encode('utf-8')
    crc_val = compute_crc32(message_bytes)
    message_with_crc = message_bytes + crc_val.to_bytes(4, 'big')

    encrypted_payload_bytes = aes_encrypt(message_with_crc, password) # salt + iv + ciphertext

    payload_bits = [int(b) for byte in encrypted_payload_bytes for b in format(byte, '08b')]
    L_payload_bits = len(payload_bits)
    logging.info(f"AES encrypted payload (salt+iv+ciphertext) converted to {L_payload_bits} bits.")

    len_bits_for_header = [int(b) for b in bin(L_payload_bits)[2:].zfill(16)] # Length of payload_bits
    len_encoded = encode_data_with_hamming(len_bits_for_header)
    logging.info(f"Payload length {L_payload_bits} (16 bits) encoded to {len(len_encoded)} Hamming bits for header.")

    payload_encoded = encode_data_with_hamming(payload_bits)
    logging.info(f"AES payload ({L_payload_bits} bits) encoded to {len(payload_encoded)} Hamming bits.")

    final_bits_for_embedding = len_encoded + payload_encoded
    return final_bits_for_embedding, L_payload_bits

def recover_message_from_extracted_payload(extracted_len_bits: list[int], extracted_payload_decoded_bits: list[int], password: str) -> str:
    """
    Recovers the message from extracted and Hamming-decoded length and payload bits.
    extracted_len_bits: 16 decoded bits representing the length of the encrypted payload.
    extracted_payload_decoded_bits: List of decoded bits for the payload (salt+iv+ciphertext).
    password: The password for decryption.
    """
    if len(extracted_len_bits) != 16:
        raise ValueError(f"Expected 16 bits for length, got {len(extracted_len_bits)}")

    L_recovered_payload_bits = int(''.join(map(str, extracted_len_bits)), 2)
    logging.info(f"Recovered payload bit length: {L_recovered_payload_bits}")

    if L_recovered_payload_bits > len(extracted_payload_decoded_bits):
        raise ValueError(f"Not enough payload bits. Expected {L_recovered_payload_bits}, got {len(extracted_payload_decoded_bits)}")

    actual_payload_bits = extracted_payload_decoded_bits[:L_recovered_payload_bits]

    if len(actual_payload_bits) % 8 != 0:
        # This should ideally not happen if L_recovered_payload_bits was correctly stored and recovered,
        # as (salt+iv+ciphertext) should be byte-aligned.
        raise ValueError(f"Recovered payload bit count {len(actual_payload_bits)} is not a multiple of 8.")

    encrypted_bytes_with_salt_iv = bytes([int(''.join(map(str, actual_payload_bits[i:i+8])), 2) for i in range(0, len(actual_payload_bits), 8)])

    decrypted_message_with_crc = aes_decrypt(encrypted_bytes_with_salt_iv, password)

    if len(decrypted_message_with_crc) < 4:
        raise ValueError("Decrypted message too short to contain CRC.")

    received_crc_bytes = decrypted_message_with_crc[-4:]
    message_bytes = decrypted_message_with_crc[:-4]

    received_crc_val = int.from_bytes(received_crc_bytes, 'big')
    computed_crc_val = compute_crc32(message_bytes)

    if received_crc_val != computed_crc_val:
        raise ValueError(f"CRC mismatch: received {received_crc_val}, computed {computed_crc_val}")

    logging.info("CRC verified successfully during recovery.")
    recovered_message_str = message_bytes.decode('utf-8')
    return recovered_message_str

def embed_bit_series(dct_blocks: list[list[np.ndarray]],
                     bits_to_embed: list[int],
                     password_str: str,  # Changed from 'password' to avoid conflict with outer scope 'key'/'password'
                     base_alpha: float,
                     K_coeff_per_bit: int,
                     embedding_pool: list[tuple],
                     num_blocks_per_channel_j: int,
                     block_variances_per_channel: list[list[float]],
                     pbar_desc: str = "Embedding bits",
                     seed_offset: int = 0):
    """Embeds a series of bits into DCT blocks. Modifies dct_blocks in-place."""
    N_pool = len(embedding_pool)
    if N_pool == 0:
        raise ValueError("Embedding pool is empty.")

    for i in tqdm(range(len(bits_to_embed)), desc=pbar_desc):
        # Seed uses the string password and current bit index (with offset)
        random.seed(password_str + str(seed_offset + i))

        # Ensure K_coeff_per_bit does not exceed available coefficients in the pool
        actual_K = min(K_coeff_per_bit, N_pool)
        idx_list_indices = random.sample(range(N_pool), actual_K) # Get indices for the pool

        p_pattern = [random.choice([1, -1]) for _ in range(actual_K)]
        current_bit_to_embed = 2 * bits_to_embed[i] - 1  # Convert bit 0/1 to -1/1

        for k_pattern_idx, pool_actual_idx in enumerate(idx_list_indices):
            bi, bj, c_idx, u, v = embedding_pool[pool_actual_idx]

            variance = block_variances_per_channel[c_idx][bi * num_blocks_per_channel_j + bj]
            adaptive_alpha = base_alpha * (1 + variance / 1000)  # Scale alpha with variance
            adaptive_alpha = min(adaptive_alpha, 1.0)  # Cap at 1.0

            dct_blocks[c_idx][bi * num_blocks_per_channel_j + bj][u, v] += \
                adaptive_alpha * current_bit_to_embed * p_pattern[k_pattern_idx]

        # Reduced verbosity for logging inside the loop or log summary afterwards
        if i % 100 == 0 or i == len(bits_to_embed) - 1: # Log every 100 bits and the last bit
             logging.info(f"{pbar_desc} - Bit {seed_offset + i}: value {bits_to_embed[i]} with adaptive_alpha up to {adaptive_alpha:.4f}, K={actual_K}")

def extract_bit_series(dct_blocks: list[list[np.ndarray]],
                       num_bits_to_extract: int,
                       password_str: str, # Changed from 'password'
                       K_coeff_per_bit: int,
                       embedding_pool: list[tuple],
                       num_blocks_per_channel_j: int,
                       pbar_desc: str = "Extracting bits",
                       seed_offset: int = 0) -> list[int]:
    """Extracts a series of bits from DCT blocks."""
    extracted_bits = []
    N_pool = len(embedding_pool)
    if N_pool == 0:
        raise ValueError("Embedding pool is empty for extraction.")

    for i in tqdm(range(num_bits_to_extract), desc=pbar_desc):
        random.seed(password_str + str(seed_offset + i))

        actual_K = min(K_coeff_per_bit, N_pool)
        idx_list_indices = random.sample(range(N_pool), actual_K)
        p_pattern = [random.choice([1, -1]) for _ in range(actual_K)]

        sum_corr = 0
        count_coeffs = 0
        for k_pattern_idx, pool_actual_idx in enumerate(idx_list_indices):
            bi, bj, c_idx, u, v = embedding_pool[pool_actual_idx]
            sum_corr += dct_blocks[c_idx][bi * num_blocks_per_channel_j + bj][u, v] * p_pattern[k_pattern_idx]
            count_coeffs +=1

        avg_corr = sum_corr / count_coeffs if count_coeffs > 0 else 0
        extracted_bit = 1 if avg_corr > 0 else 0
        extracted_bits.append(extracted_bit)

        if i % 100 == 0 or i == num_bits_to_extract - 1: # Log every 100 bits and the last bit
            logging.info(f"{pbar_desc} - Bit {seed_offset + i}: extracted value {extracted_bit}, correlation {avg_corr:.4f}, K={actual_K}")

    return extracted_bits

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
    b_padded, g_padded, r_padded = cv2.split(img) # These are already padded
    padded_channels = [b_padded, g_padded, r_padded]

    h_padded, w_padded = img.shape[:2] # Dimensions after padding
    num_blocks_i, num_blocks_j = h_padded // 8, w_padded // 8

    # Calculate block variances from original (padded) spatial domain blocks
    block_variances = [[] for _ in range(3)]
    for c_idx, channel_data in enumerate(padded_channels):
        for bi in range(num_blocks_i):
            for bj in range(num_blocks_j):
                block = channel_data[bi*8:(bi+1)*8, bj*8:(bj+1)*8]
                block_variances[c_idx].append(compute_block_variance(block))

    # Apply DCT to padded image channels
    dct_blocks = apply_dct_to_image_channels(padded_channels) # dct_blocks is [ch1_blocks, ch2_blocks, ch3_blocks]
    logging.info(f"DCT computed for {len(dct_blocks[0]) if dct_blocks and dct_blocks[0] else 0} blocks per channel")

    # Select mid-frequency coefficients
    selected_uv = [(u, v) for u in range(1, 8) for v in range(1, 8)]
    pool = [(bi, bj, c_idx, u, v) for bi in range(num_blocks_i)
            for bj in range(num_blocks_j) for c_idx in range(3) for (u, v) in selected_uv]
    N = len(pool)
    logging.info(f"Coefficient pool size: {N}")


    # Prepare the full bit sequence to embed (header + payload)
    # The key (int) is converted to str for use as a password
    encoded_bits, L_payload_bits = prepare_message_for_embedding(message, str(key))
    M = len(encoded_bits) # M is now the total number of Hamming-encoded bits to embed

    # L_payload_bits is the length of (salt+iv+ciphertext) in bits. This is what the header stores.
    logging.info(f"Total Hamming-encoded bits to embed: {M}. Payload bit length (L): {L_payload_bits}.") # M is len(encoded_bits)

    # Embed the combined encoded bits (header + payload)
    embed_bit_series(
        dct_blocks=dct_blocks,
        bits_to_embed=encoded_bits,
        password_str=str(key), # Use string representation of the integer key as password for seeding
        base_alpha=alpha,
        K_coeff_per_bit=K,
        embedding_pool=pool,
        num_blocks_per_channel_j=num_blocks_j, # w_padded // 8
        block_variances_per_channel=block_variances,
        pbar_desc="Embedding all bits",
        seed_offset=0
    )

    # Reconstruct image from modified DCT blocks
    # dct_blocks is already the modified list of lists of DCT blocks
    watermarked_channels = apply_idct_to_dct_blocks(dct_blocks, h_padded, w_padded)
    watermarked_img = cv2.merge(watermarked_channels) # Channels are already uint8 and clipped
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
    b_padded, g_padded, r_padded = cv2.split(img) # These are already padded
    padded_channels = [b_padded, g_padded, r_padded]

    h_padded, w_padded = img.shape[:2] # Dimensions after padding
    num_blocks_i, num_blocks_j = h_padded // 8, w_padded // 8

    # Apply DCT to padded image channels
    dct_blocks = apply_dct_to_image_channels(padded_channels) # dct_blocks is [ch1_blocks, ch2_blocks, ch3_blocks]
    logging.info(f"DCT computed for {len(dct_blocks[0]) if dct_blocks and dct_blocks[0] else 0} blocks per channel for extraction")

    # Define coefficient pool
    selected_uv = [(u, v) for u in range(1, 8) for v in range(1, 8)]
    pool = [(bi, bj, c_idx, u, v) for bi in range(num_blocks_i)
            for bj in range(num_blocks_j) for c_idx in range(3) for (u, v) in selected_uv]
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
        # Extract 28 raw bits for the length header
        len_encoded_extracted = extract_bit_series(
            dct_blocks=dct_blocks,
            num_bits_to_extract=28,
            password_str=str(key),
            K_coeff_per_bit=current_K,
            embedding_pool=pool,
            num_blocks_per_channel_j=num_blocks_j, # w_padded // 8
            pbar_desc=f"Extracting length (attempt {attempt+1}, K={current_K})",
            seed_offset=0
        )

        # Decode length
        # Ensure we only try to decode if we have enough bits for the length field (28 bits)
        if len(len_encoded_extracted) < 28:
            logging.error(f"Not enough bits extracted for length field: got {len(len_encoded_extracted)}, expected 28.")
            attempt +=1
            current_alpha = min(current_alpha * 2, max_alpha)
            current_K = min(current_K * 2, N)
            logging.info(f"Attempt {attempt} (length decode) failed, adjusting alpha to {current_alpha}, K to {current_K}")
            continue # Skip to next attempt

        len_bits_from_header_decoded = decode_data_with_hamming(len_encoded_extracted[:28]) # Process only the 28 bits for length
        L_payload_bits_from_header = int(''.join(map(str, len_bits_from_header_decoded[:16])), 2) # Length is 16 bits
        logging.info(f"Extracted payload bit length (from header): {L_payload_bits_from_header}")

        # Validate length: L_payload_bits_from_header is length of (salt+iv+ciphertext) in bits.
        # Max message 128 chars -> ~512 bytes. CRC 4 bytes. Total ~516. AES pads to multiple of 16. E.g. 528 bytes.
        # Salt (16) + IV (16) + Ciphertext (e.g. 528) = 560 bytes = 4480 bits.
        # Max reasonable length can be set, e.g., 1024 bytes for salt+iv+ciphertext = 8192 bits.
        # Must be multiple of 16 bytes (128 bits) because salt+iv is 32 bytes and AES output is block-aligned.
        if L_payload_bits_from_header > 0 and L_payload_bits_from_header <= 8192 and L_payload_bits_from_header % 128 == 0:
            break # Valid length found

        logging.warning(f"Invalid L_payload_bits_from_header: {L_payload_bits_from_header}. Not in range (0, 8192] or not multiple of 128.")
        attempt += 1
        current_alpha = min(current_alpha * 2, max_alpha)
        current_K = min(current_K * 2, N)
        logging.info(f"Attempt {attempt} (length validation) failed, adjusting alpha to {current_alpha}, K to {current_K}")

    if attempt >= max_attempts:
        raise ValueError("Failed to extract valid payload length from header after maximum attempts")

    # Determine how many bits to extract for the Hamming-encoded payload
    # Number of 4-bit data chunks in payload = (L_payload_bits_from_header + 3) // 4
    # Number of 7-bit Hamming chunks for payload = (L_payload_bits_from_header + 3) // 4
    num_payload_data_chunks = (L_payload_bits_from_header + 3) // 4
    M_payload_encoded_bits = num_payload_data_chunks * 7

    logging.info(f"Expecting {M_payload_encoded_bits} Hamming-encoded bits for the payload (derived from L={L_payload_bits_from_header}).")

    # Extract the Hamming-encoded payload bits
    raw_payload_bits_extracted = []
    # Reset attempt counter and parameters for payload extraction phase if needed, or continue with current
    # For simplicity, let's reset attempts for this critical phase.
    attempt = 0 # Reset for payload extraction attempts
    # current_alpha and current_K could be reset or tuned separately for payload

    while attempt < max_attempts:
        # Extract M_payload_encoded_bits for the payload
        raw_payload_bits_extracted = extract_bit_series(
            dct_blocks=dct_blocks,
            num_bits_to_extract=M_payload_encoded_bits,
            password_str=str(key),
            K_coeff_per_bit=current_K,
            embedding_pool=pool,
            num_blocks_per_channel_j=num_blocks_j, # w_padded // 8
            pbar_desc=f"Extracting payload (attempt {attempt+1}, K={current_K})",
            seed_offset=28 # Start seed from 28, as first 28 were for header
        )
        logging.info(f"Extracted {len(raw_payload_bits_extracted)} raw bits for payload in attempt {attempt+1}.")

        if len(raw_payload_bits_extracted) != M_payload_encoded_bits:
            logging.warning(f"Extracted bit count {len(raw_payload_bits_extracted)} for payload does not match expected {M_payload_encoded_bits}.")
            # This case might require rethinking if K needs to be drastically different for payload
            attempt += 1
            current_alpha = min(current_alpha * 2, max_alpha) # Example: Adjust alpha
            logging.info(f"Attempt {attempt} (payload extraction bit count) failed, adjusting alpha to {current_alpha}, K to {current_K}")
            if attempt >=max_attempts: raise ValueError("Failed to extract correct number of payload bits.")
            continue


        # Decode the raw payload bits (Hamming decoding)
        try:
            if len(raw_payload_bits_extracted) % 7 != 0:
                 logging.warning(f"Length of raw extracted payload bits ({len(raw_payload_bits_extracted)}) is not a multiple of 7. Truncating.")
                 raw_payload_bits_extracted = raw_payload_bits_extracted[:-(len(raw_payload_bits_extracted)%7)]

            if not raw_payload_bits_extracted:
                raise ValueError("No payload bits to decode after truncation.")

            decoded_payload_bits_full = decode_data_with_hamming(raw_payload_bits_extracted)

            # Now, recover the message using the decoded length (from header) and decoded payload
            # The password (str(key)) is passed for decryption.
            # len_bits_from_header_decoded are the 16 bits for L_payload_bits_from_header
            final_message = recover_message_from_extracted_payload(len_bits_from_header_decoded, decoded_payload_bits_full, str(key))
            return final_message # Success

        except ValueError as e: # Catches CRC mismatch, decoding errors, length issues from recover_message
            logging.error(f"Message recovery failed: {str(e)}")
            attempt += 1
            # Adjust parameters for next attempt
            current_alpha = min(current_alpha * 2, max_alpha)
            current_K = min(current_K * 2, N) # Or other strategy
            logging.info(f"Attempt {attempt} (message recovery) failed, adjusting alpha to {current_alpha}, K to {current_K}")
            if attempt >= max_attempts:
                 raise ValueError(f"Failed to recover message after maximum attempts: {str(e)}")
        except Exception as e: # Catch any other unexpected error during recovery
            logging.error(f"Unexpected error during message recovery: {str(e)}")
            raise # Re-throw if it's not a planned retryable error

    raise ValueError("Failed to extract valid message after maximum attempts (payload phase).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Advanced DCT-based Spread Spectrum Watermarking Tool")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Embed command
    embed_parser = subparsers.add_parser("embed", help="Embed a message into a color image")
    embed_parser.add_argument("--image", required=True, help="Input color image path (PNG/JPEG)")
    embed_parser.add_argument("--message", required=True, help="Message to embed (max 128 chars)")
    embed_parser.add_argument("--output", required=True, help="Output watermarked color image path")
    embed_parser.add_argument("--key", type=int, required=True, help="Integer key for PRNG seed and basis for encryption password")
    embed_parser.add_argument("--alpha", type=float, default=1.0, help="Initial embedding strength (default: 1.0)")
    embed_parser.add_argument("--K", type=int, default=500, help="Initial coefficients per bit (default: 500)")

    # Extract command
    extract_parser = subparsers.add_parser("extract", help="Extract a message from a color image")
    extract_parser.add_argument("--image", required=True, help="Input watermarked color image path (PNG/JPEG)")
    extract_parser.add_argument("--key", type=int, required=True, help="Integer key used during embedding and basis for decryption password")
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