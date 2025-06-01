# --- Standard Python Libraries ---
import argparse
import hashlib
import logging
import os
import random
import sys
import zlib
from pathlib import Path
from typing import Any, List, Optional, Set, Tuple

# --- Third-Party Libraries ---
import cv2
import numpy as np
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from tqdm import tqdm

# --- Local Application Imports ---
# (None currently)

# --- Constants ---
# Logging Configuration
LOG_FILE_NAME: str = 'watermarking.log'
LOG_FORMAT: str = '%(asctime)s - %(levelname)s - %(message)s'

# Hamming Code Parameters
HAMMING_DATA_BITS: int = 4
HAMMING_CODEWORD_BITS: int = 7
HAMMING_PARITY_BITS: int = HAMMING_CODEWORD_BITS - HAMMING_DATA_BITS

# AES Encryption Parameters
AES_KEY_SIZE_BYTES: int = 32
AES_IV_SIZE_BYTES: int = 16
AES_BLOCK_SIZE_BYTES: int = 16 # AES block size, also used for padding

# CRC Parameters
CRC_SIZE_BYTES: int = 4

# Image Processing Parameters
IMAGE_BLOCK_SIZE: int = 8
IMAGE_CHANNELS: int = 3 # Expected number of color channels
DCT_COEFF_START_INDEX: int = 1 # Start index for selecting u,v in DCT block (excluding DC)
DCT_COEFF_END_INDEX: int = 8   # End index for selecting u,v (exclusive, so up to 7)

# Message Constraints
MAX_MESSAGE_CHARS: int = 128

# Embedding & Extraction Parameters
DEFAULT_ALPHA: float = 1.0
MAX_ALPHA: float = 1.0
DEFAULT_K_COEFFS: int = 500
MAX_EXTRACTION_ATTEMPTS: int = 5
# INITIAL_EXTRACTION_ALPHA: float = 0.1 # This constant is not actively used.
VARIANCE_SCALING_FACTOR: float = 1000.0 # For adaptive alpha

# Length Field Parameters
LENGTH_FIELD_BITS: int = 16 # Number of bits to represent L (length of message after AES/CRC/Padding)
BITS_IN_BYTE: int = 8 # Number of bits in a byte
MAX_L_VALUE: int = (2**LENGTH_FIELD_BITS) -1 # Max value L can hold
EXPECTED_L_MULTIPLE_BITS: int = AES_BLOCK_SIZE_BYTES * BITS_IN_BYTE # L should be multiple of this (IV + ciphertext block)
ENCODED_LENGTH_BITS: int = (LENGTH_FIELD_BITS // HAMMING_DATA_BITS) * HAMMING_CODEWORD_BITS # e.g., 16/4 * 7 = 28 bits

# File Handling
VALID_IMAGE_EXTENSIONS: Set[str] = {".png", ".jpg", ".jpeg"}

# CLI Command Names
CMD_EMBED: str = "embed"
CMD_EXTRACT: str = "extract"

# Logging and Retry Behavior
LOG_INTERVAL_EMBED_BITS: int = 100
LOG_INTERVAL_EXTRACT_LENGTH_BITS: int = 10
LOG_INTERVAL_EXTRACT_MESSAGE_BITS: int = 100
K_RETRY_MULTIPLIER: int = 2

# Character Encoding
PRINTABLE_ASCII_MIN: int = 32
PRINTABLE_ASCII_MAX: int = 127 # Max value for ord(char) to be considered printable (exclusive)
# BITS_IN_BYTE is defined above in Length Field Parameters

# PBKDF2 for Key Derivation
PBKDF2_SALT: bytes = b'_w4t3rm4rk1ngS4lt_' # Example 16-byte salt for AES key
PBKDF2_SALT_RNG_SEED: bytes = b's33d_f0r_r4nd0mn3ss' # Different salt for PRNG seed derivation
PBKDF2_ITERATIONS: int = 100000 # Number of iterations for PBKDF2

# Cryptography Helpers
PBKDF2_HASH_ALGORITHM: str = 'sha256'
DEFAULT_ENCODING: str = 'utf-8'
BYTE_ORDER_BIG: str = 'big' # For to_bytes and from_bytes
CRC32_MASK: int = 0xFFFFFFFF # To ensure positive value, consistent across Python versions

# PRNG
PRNG_SEED_DERIVATION_LENGTH_BYTES: int = 16 # For deriving the seed for random.Random

# Bit/Byte Operations
BINARY_FORMAT_BYTE: str = '08b' # For format(byte, '08b')
BINARY_PREFIX_LEN: int = len('0b') # For slicing "0b" from bin() output, e.g. bin(L)[BINARY_PREFIX_LEN:]
PADDING_BIT: int = 0 # For padding bit lists

# Image Processing
PIXEL_VALUE_MIN: int = 0
PIXEL_VALUE_MAX: int = 255

# Bipolar Operations (for spread spectrum modulation)
BIPOLAR_MAPPING_VALUES: List[int] = [-1, 1] # For rng.choice([-1, 1])
BIPOLAR_CONVERSION_MULTIPLIER: int = 2
BIPOLAR_CONVERSION_SUBTRACTOR: int = 1

# Extraction Logic
INITIAL_INVALID_LENGTH: int = -1 # Initial value for extracted length
BINARY_BASE: int = 2 # For int(..., 2) for binary string to int conversion

# Logging, CLI, and Misc
LOG_PREVIEW_LENGTH: int = 50 # For previewing extracted messages in logs
ERROR_EXIT_CODE: int = 1
ALLOWED_WHITESPACE_CHARS: str = '\n\r\t' # For validating message characters


# --- Custom Exception Classes ---
class WatermarkingError(Exception):
    """Base class for exceptions in this module."""
    pass

class ConfigurationError(WatermarkingError):
    """For errors in configuration or input parameters."""
    pass

class ImageProcessingError(WatermarkingError):
    """For errors related to image loading, format, or manipulation."""
    pass

class EmbeddingError(WatermarkingError):
    """For errors specific to the embedding process."""
    pass

class ExtractionError(WatermarkingError):
    """For errors specific to the extraction process."""
    pass

class CryptoError(WatermarkingError):
    """For errors related to cryptographic operations."""
    pass


# Configure logging
logging.basicConfig(
    filename=LOG_FILE_NAME,
    level=logging.INFO,
    format=LOG_FORMAT
)

# --- Hamming Code Functions ---

def hamming_encode(data_bits: List[int]) -> List[int]:
    """
    Encode 4 data bits into 7 bits using Hamming(7,4) code.
    This implementation is specific to Hamming(7,4).

    Args:
        data_bits: A list of 4 integers (0 or 1) representing the data bits.

    Returns:
        A list of 7 integers (0 or 1) representing the Hamming codeword.

    Raises:
        AssertionError: If the input is not a list of 4 bits.
    """
    assert len(data_bits) == HAMMING_DATA_BITS, f"Input must be {HAMMING_DATA_BITS} bits"
    d1, d2, d3, d4 = data_bits
    p1 = d1 ^ d2 ^ d4
    p2 = d1 ^ d3 ^ d4
    p3 = d2 ^ d3 ^ d4
    return [p1, p2, d1, p3, d2, d3, d4]

def hamming_decode(codeword: List[int]) -> List[int]:
    """
    Decode a 7-bit Hamming codeword to 4 data bits, correcting single-bit errors.
    This implementation is specific to Hamming(7,4).

    Args:
        codeword: A list of 7 integers (0 or 1) representing the codeword.

    Returns:
        A list of 4 integers (0 or 1) representing the decoded data bits.

    Raises:
        AssertionError: If the input is not a list of 7 bits.
    """
    assert len(codeword) == HAMMING_CODEWORD_BITS, f"Input must be {HAMMING_CODEWORD_BITS} bits"
    p1, p2, d1, p3, d2, d3, d4 = codeword # Specific to (7,4) structure
    s1 = p1 ^ d1 ^ d2 ^ d4
    s2 = p2 ^ d1 ^ d3 ^ d4
    s3 = p3 ^ d2 ^ d3 ^ d4
    syndrome = s1 + 2 * s2 + 4 * s3 # Specific to (7,4) error positions
    if syndrome != 0 and syndrome <= HAMMING_CODEWORD_BITS: # Check syndrome is valid
        error_pos = syndrome - 1
        codeword[error_pos] ^= 1
    # Return data bits, specific to (7,4) structure where these are data bit positions
    return [codeword[2], codeword[4], codeword[5], codeword[6]]

# --- Cryptographic Functions ---

def derive_aes_key(passphrase: str, salt: bytes, key_length: int = AES_KEY_SIZE_BYTES) -> bytes:
    """
    Derives a cryptographic key of specified length from a passphrase and salt
    using PBKDF2-HMAC-SHA256.

    Args:
        passphrase: The input passphrase (expected to be a string).
        salt: A salt value (bytes).
        key_length: The desired length of the derived key in bytes.
                    Defaults to AES_KEY_SIZE_BYTES.

    Returns:
        The derived key as bytes.
    """
    key: bytes = hashlib.pbkdf2_hmac(
        PBKDF2_HASH_ALGORITHM,
        passphrase.encode(DEFAULT_ENCODING), # Ensure passphrase is utf-8 encoded for PBKDF2
        salt,
        PBKDF2_ITERATIONS,
        dklen=key_length
    )
    return key

def aes_encrypt(message_bytes: bytes, passphrase: str) -> bytes:
    """
    Encrypts message bytes using AES-256-CBC with a key derived from the passphrase.
    The IV is randomly generated and prepended to the ciphertext.

    Args:
        message_bytes: The raw bytes of the message to encrypt.
        passphrase: The passphrase used to derive the encryption key.

    Returns:
        The encrypted data as bytes, with the IV prepended (IV + Ciphertext).
    """
    key_bytes: bytes = derive_aes_key(passphrase, PBKDF2_SALT, AES_KEY_SIZE_BYTES)
    iv: bytes = os.urandom(AES_IV_SIZE_BYTES)
    cipher: Cipher = Cipher(algorithms.AES(key_bytes), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    # Pad message to multiple of AES_BLOCK_SIZE_BYTES
    pad_length = AES_BLOCK_SIZE_BYTES - (len(message_bytes) % AES_BLOCK_SIZE_BYTES)
    padded_message = message_bytes + bytes([pad_length] * pad_length)
    encrypted = encryptor.update(padded_message) + encryptor.finalize()
    logging.info(f"Message encrypted with AES-{AES_KEY_SIZE_BYTES*BITS_IN_BYTE}, length: {len(encrypted)} bytes (excluding IV)")
    return iv + encrypted  # Prepend IV for decryption

def aes_decrypt(encrypted_bytes_with_iv: bytes, passphrase: str) -> bytes:
    """
    Decrypts AES-256-CBC encrypted bytes (IV prepended) with a key derived from the passphrase.

    Args:
        encrypted_bytes_with_iv: The encrypted data, with IV prepended.
        passphrase: The passphrase used to derive the decryption key.

    Returns:
        The decrypted message as bytes.

    Raises:
        CryptoError: If decryption or unpadding fails.
    """
    key_bytes: bytes = derive_aes_key(passphrase, PBKDF2_SALT, AES_KEY_SIZE_BYTES)
    iv: bytes = encrypted_bytes_with_iv[:AES_IV_SIZE_BYTES]
    ciphertext: bytes = encrypted_bytes_with_iv[AES_IV_SIZE_BYTES:]
    cipher: Cipher = Cipher(algorithms.AES(key_bytes), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    padded_message = decryptor.update(ciphertext) + decryptor.finalize()

    try:
        pad_value = padded_message[-1]
        if not (0 < pad_value <= AES_BLOCK_SIZE_BYTES):
            raise CryptoError(f"Invalid padding value during decryption: {pad_value}. Must be between 1 and {AES_BLOCK_SIZE_BYTES}.")

        # Check if message is long enough for the declared padding
        if len(padded_message) < pad_value:
            raise CryptoError(f"Padded message is too short (length {len(padded_message)}) for declared padding value ({pad_value}).")

        message = padded_message[:-pad_value]
    except IndexError:
        # This occurs if padded_message is empty or too short to access padded_message[-1] or apply slicing.
        raise CryptoError("Error processing padding: Padded message is unexpectedly short.")
    except CryptoError: # Re-raise CryptoErrors from above checks
        raise
    except Exception as e: # Catch any other unexpected error during padding removal
        raise CryptoError(f"An unexpected error occurred during AES unpadding: {str(e)}")

    logging.info(f"Message decrypted with AES-{AES_KEY_SIZE_BYTES*BITS_IN_BYTE}, length: {len(message)} bytes")
    return message

def compute_crc32(data: bytes) -> int:
    """
    Computes the CRC-32 checksum for the given data.

    Args:
        data: The input data as bytes.

    Returns:
        The CRC-32 checksum as a 32-bit unsigned integer.
    """
    return zlib.crc32(data) & CRC32_MASK # Ensure positive value, consistent across Python versions

# --- Helper Functions for Image Processing ---

def load_and_pad_image(image_path_str: str) -> Tuple[np.ndarray, int, int, int, int]:
    """
    Loads an image, validates its format and properties, and pads its dimensions
    to be multiples of IMAGE_BLOCK_SIZE.

    Args:
        image_path_str: Path to the image file.

    Returns:
        A tuple containing:
            - img_padded (np.ndarray): The loaded and padded image.
            - h_orig (int): Original height of the image.
            - w_orig (int): Original width of the image.
            - h_pad (int): Padded height of the image.
            - w_pad (int): Padded width of the image.

    Raises:
        ImageProcessingError: If the image cannot be loaded, is not a valid format,
                              or does not meet channel requirements.
    """
    image_path: Path = Path(image_path_str)
    if not image_path.is_file():
        raise ImageProcessingError(f"Input image path does not exist or is not a file: '{image_path_str}'")
    if image_path.suffix.lower() not in VALID_IMAGE_EXTENSIONS:
        raise ImageProcessingError(f"Image must be one of {VALID_IMAGE_EXTENSIONS}, got '{image_path.suffix}'")

    img: Optional[np.ndarray] = cv2.imread(image_path_str, cv2.IMREAD_COLOR)
    if img is None:
        raise ImageProcessingError(f"Cannot load image at '{image_path_str}'. Check file integrity and path.")
    if len(img.shape) != 3 or img.shape[2] != IMAGE_CHANNELS:
        raise ImageProcessingError(f"Input image must be a color image with {IMAGE_CHANNELS} channels, shape found: {img.shape}")

    h_orig: int = img.shape[0]
    w_orig: int = img.shape[1]
    
    pad_h: int = (IMAGE_BLOCK_SIZE - h_orig % IMAGE_BLOCK_SIZE) % IMAGE_BLOCK_SIZE
    pad_w: int = (IMAGE_BLOCK_SIZE - w_orig % IMAGE_BLOCK_SIZE) % IMAGE_BLOCK_SIZE
    
    img_padded: np.ndarray = img
    if pad_h > 0 or pad_w > 0:
        img_padded = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)

    h_pad: int = img_padded.shape[0]
    w_pad: int = img_padded.shape[1]
    logging.info(f"Image loaded: original {h_orig}x{w_orig}, padded to {h_pad}x{w_pad}. Blocks: {h_pad//IMAGE_BLOCK_SIZE}x{w_pad//IMAGE_BLOCK_SIZE}")
    return img_padded, h_orig, w_orig, h_pad, w_pad

def compute_dct_blocks(
    image_channels: List[np.ndarray],
    h_pad: int,
    w_pad: int,
    compute_variances: bool = False
) -> Tuple[List[List[np.ndarray]], Optional[List[List[float]]]]:
    """
    Computes DCT (Discrete Cosine Transform) blocks for each channel of an image.
    Optionally computes the variance of each block.

    Args:
        image_channels: A list of numpy arrays, where each array is a color channel of the (padded) image.
        h_pad: Padded height of the image.
        w_pad: Padded width of the image.
        compute_variances: If True, computes and returns block variances.

    Returns:
        A tuple containing:
            - dct_blocks_all_channels (List[List[np.ndarray]]): A list of lists, where each inner list
              contains the DCT blocks (as float32 numpy arrays) for a channel.
            - block_variances_all_channels (Optional[List[List[float]]]): A list of lists for block variances
              if compute_variances is True, otherwise None.
    """
    num_blocks_i: int = h_pad // IMAGE_BLOCK_SIZE
    num_blocks_j: int = w_pad // IMAGE_BLOCK_SIZE

    dct_blocks_all_channels: List[List[np.ndarray]] = [[] for _ in range(IMAGE_CHANNELS)]
    block_variances_all_channels: Optional[List[List[float]]] = [[] for _ in range(IMAGE_CHANNELS)] if compute_variances else None

    for c_idx in range(IMAGE_CHANNELS):
        for bi in range(num_blocks_i):
            for bj in range(num_blocks_j):
                block: np.ndarray = image_channels[c_idx][
                    bi*IMAGE_BLOCK_SIZE:(bi+1)*IMAGE_BLOCK_SIZE,
                    bj*IMAGE_BLOCK_SIZE:(bj+1)*IMAGE_BLOCK_SIZE
                ]
                dct_blocks_all_channels[c_idx].append(cv2.dct(block.astype(np.float32)))
                if compute_variances and block_variances_all_channels is not None: # Ensure mypy knows it's not None
                    block_variances_all_channels[c_idx].append(np.var(block))

    logging.info(f"DCT computed for {len(dct_blocks_all_channels[0])} blocks per channel.")
    return dct_blocks_all_channels, block_variances_all_channels

def reconstruct_image_from_dct_blocks(
    dct_blocks_all_channels: List[List[np.ndarray]],
    h_pad: int,
    w_pad: int,
    h_orig: int,
    w_orig: int
) -> np.ndarray:
    """
    Reconstructs an image from its DCT blocks, applies clipping, converts to uint8,
    and removes padding to restore original dimensions.

    Args:
        dct_blocks_all_channels: A list of lists, where each inner list contains the
                                 DCT blocks for a channel.
        h_pad: Padded height of the image from which DCT blocks were derived.
        w_pad: Padded width.
        h_orig: Original height of the image (pre-padding).
        w_orig: Original width of the image.

    Returns:
        The reconstructed image as a numpy array (uint8).
    """
    num_blocks_i: int = h_pad // IMAGE_BLOCK_SIZE
    num_blocks_j: int = w_pad // IMAGE_BLOCK_SIZE
    
    reconstructed_channels: List[np.ndarray] = [np.zeros((h_pad, w_pad), dtype=np.float32) for _ in range(IMAGE_CHANNELS)]

    for c_idx in range(IMAGE_CHANNELS):
        for bi in range(num_blocks_i):
            for bj in range(num_blocks_j):
                dct_block_index = bi * num_blocks_j + bj
                idct_block = cv2.idct(dct_blocks_all_channels[c_idx][dct_block_index])
                reconstructed_channels[c_idx][
                    bi*IMAGE_BLOCK_SIZE:(bi+1)*IMAGE_BLOCK_SIZE,
                    bj*IMAGE_BLOCK_SIZE:(bj+1)*IMAGE_BLOCK_SIZE
                ] = idct_block
    
    # Use np.rint for rounding, then clip and convert type
    merged_padded_image = cv2.merge([
        np.rint(channel).clip(PIXEL_VALUE_MIN, PIXEL_VALUE_MAX).astype(np.uint8) for channel in reconstructed_channels
    ])
    
    # Remove padding to get the original image dimensions
    final_image: np.ndarray = merged_padded_image[:h_orig, :w_orig]
    return final_image

# --- Coefficient Pool Generation ---
def build_coefficient_pool(num_blocks_i: int, num_blocks_j: int, num_channels: int = IMAGE_CHANNELS) -> List[Tuple[int, int, int, int, int]]:
    """
    Generates a list of all available (block_row, block_col, channel_idx, u, v) coordinates
    for DCT coefficient modification, excluding the DC component.

    Args:
        num_blocks_i: Number of blocks in the i-direction (height).
        num_blocks_j: Number of blocks in the j-direction (width).
        num_channels: Number of color channels in the image. Defaults to IMAGE_CHANNELS.

    Returns:
        A list of tuples, where each tuple represents a selectable DCT coefficient coordinate.
    """
    selected_uv: List[Tuple[int, int]] = [
        (u, v) for u in range(DCT_COEFF_START_INDEX, DCT_COEFF_END_INDEX)
        for v in range(DCT_COEFF_START_INDEX, DCT_COEFF_END_INDEX)
    ]

    pool: List[Tuple[int, int, int, int, int]] = [
        (bi, bj, c_idx, u, v)
        for bi in range(num_blocks_i)
        for bj in range(num_blocks_j)
        for c_idx in range(num_channels)
        for (u, v) in selected_uv
    ]
    return pool

# --- Main Embedding and Extraction Functions ---

def embed(image_path_str: str, message: str, output_path_str: str, key: str, alpha: float, K: int) -> None:
    """
    Embeds a secret message into a color image using a DCT-based spread spectrum technique
    with error correction (Hamming codes) and AES encryption.

    Args:
        image_path_str: Path to the input color image (PNG or JPEG).
        message: The secret message string to embed (max MAX_MESSAGE_CHARS).
        output_path_str: Path to save the watermarked output image.
        key: Passphrase (string) for AES key derivation and PRNG seeding.
        alpha: Initial embedding strength (adaptive based on block variance).
        K: Number of DCT coefficients to modify per bit of the encoded message.

    Raises:
        ConfigurationError: For invalid input parameters.
        ImageProcessingError: For issues with image loading or format.
        EmbeddingError: For errors during the embedding process (e.g., capacity issues).
        CryptoError: For errors during cryptographic operations.
    """
    # --- Input Validation ---
    if not (0 < alpha <= MAX_ALPHA):
        raise ConfigurationError(f"Alpha value {alpha} is out of valid range (0, {MAX_ALPHA}].")
    if K <= 0:
        raise ConfigurationError(f"K value {K} must be positive.")
    if not message: # Check for empty message
        raise ConfigurationError("Message cannot be empty.")
    if len(message) > MAX_MESSAGE_CHARS: # This is already checked above.
        raise ConfigurationError(f"Message too long; maximum {MAX_MESSAGE_CHARS} characters, got {len(message)}.")
    if image_path_str == output_path_str:
        raise ConfigurationError(f"Input image path ('{image_path_str}') and output image path ('{output_path_str}') cannot be the same. Please specify a different output path.")

    output_path = Path(output_path_str) # output_path is defined here but image_path was part of load_and_pad

    # Load, validate, and pad the image
    img_padded, h_orig, w_orig, h_pad, w_pad = load_and_pad_image(image_path_str)

    # Split into channels
    b, g, r = cv2.split(img_padded)
    channels_data = [b, g, r] # Padded channels

    # Compute DCT blocks and variances
    num_blocks_i: int = h_pad // IMAGE_BLOCK_SIZE # Type hint added for clarity
    num_blocks_j: int = w_pad // IMAGE_BLOCK_SIZE # Type hint added for clarity
    dct_blocks: List[List[np.ndarray]] # Type hint added
    block_variances: Optional[List[List[float]]] # Type hint added
    dct_blocks, block_variances = compute_dct_blocks(channels_data, h_pad, w_pad, compute_variances=True)

    if block_variances is None: # Should not happen if compute_variances=True
        raise EmbeddingError("Block variances were not computed, which are required for adaptive embedding.")

    # Build coefficient pool
    coeff_pool: List[Tuple[int,int,int,int,int]] = build_coefficient_pool(num_blocks_i, num_blocks_j, IMAGE_CHANNELS)
    N_coeffs_total: int = len(coeff_pool)
    logging.info(f"Coefficient pool size: {N_coeffs_total}")

    message_bytes: bytes = message.encode(DEFAULT_ENCODING)
    crc: int = compute_crc32(message_bytes)
    message_with_crc: bytes = message_bytes + crc.to_bytes(CRC_SIZE_BYTES, BYTE_ORDER_BIG)
    encrypted_bytes_with_iv: bytes = aes_encrypt(message_with_crc, key) # key is now str

    # msg_bits are bits of (IV + Ciphertext)
    msg_bits = [int(b) for byte in encrypted_bytes_with_iv for b in format(byte, BINARY_FORMAT_BYTE)]
    L = len(msg_bits)
    if L > MAX_L_VALUE:
        raise EmbeddingError(f"Encrypted message bit length ({L}) exceeds maximum representable by LENGTH_FIELD_BITS ({MAX_L_VALUE})")
    if L % EXPECTED_L_MULTIPLE_BITS != 0:
        # This should not happen if AES encryption works correctly (IV + multiple of blocksize for ciphertext)
        raise EmbeddingError(f"Encrypted message bit length ({L}) is not a multiple of {EXPECTED_L_MULTIPLE_BITS} bits. This may indicate an internal issue with AES padding or IV handling.")

    logging.info(f"Message (with CRC) encrypted and converted to {L} bits (IV + Ciphertext)")

    len_bits_payload = [int(b) for b in bin(L)[BINARY_PREFIX_LEN:].zfill(LENGTH_FIELD_BITS)]
    len_encoded_hamming = []
    for i in range(0, LENGTH_FIELD_BITS, HAMMING_DATA_BITS):
        chunk = len_bits_payload[i:i+HAMMING_DATA_BITS]
        len_encoded_hamming.extend(hamming_encode(chunk))
    # ENCODED_LENGTH_BITS should match len(len_encoded_hamming)
    logging.info(f"Length {L} encoded to {len(len_encoded_hamming)} bits using Hamming({HAMMING_CODEWORD_BITS},{HAMMING_DATA_BITS})")

    msg_encoded_hamming = []
    # Pad msg_bits to be multiple of HAMMING_DATA_BITS before encoding
    num_data_chunks = (L + HAMMING_DATA_BITS - 1) // HAMMING_DATA_BITS
    padded_msg_bits = msg_bits + [PADDING_BIT] * (num_data_chunks * HAMMING_DATA_BITS - L)

    for i in range(0, len(padded_msg_bits), HAMMING_DATA_BITS):
        chunk = padded_msg_bits[i:i+HAMMING_DATA_BITS]
        msg_encoded_hamming.extend(hamming_encode(chunk))
    logging.info(f"Message bits ({L}) padded and encoded to {len(msg_encoded_hamming)} bits using Hamming({HAMMING_CODEWORD_BITS},{HAMMING_DATA_BITS})")

    encoded_bits_total = len_encoded_hamming + msg_encoded_hamming
    M_total_bits = len(encoded_bits_total)
    logging.info(f"Total encoded bits to embed (Length + Message): {M_total_bits}")

    # Calculate maximum embeddable bits more accurately
    # Each bit is embedded into 'actual_K' coefficients.
    # So, N_coeffs_total can support N_coeffs_total / actual_K bits if K is fixed.
    # However, K in args is a target, actual_K can be min(K, N_coeffs_total)
    # A simpler check: if M_total_bits > N_coeffs_total, it's impossible even if K=1 for all bits.
    if M_total_bits == 0: # Should not happen with valid message
        raise EmbeddingError("Total bits to embed is zero. Check message and encoding.")
    if N_coeffs_total == 0:
        raise EmbeddingError("Coefficient pool is empty. Cannot embed into this image (e.g. too small or no suitable AC coeffs).")

    # Max capacity: each coefficient can hold one bit independently (if K=1).
    # If K > 1, then K coefficients are used for one bit.
    # This check was M_total_bits > N_coeffs_total * K. This seems to imply N_coeffs_total * K is the capacity.
    # It should rather be N_coeffs_total / K if we need K distinct coeffs for each bit.
    # Or, if coefficients can be reused (which they are, via random.sample with a seed per bit),
    # the capacity is more complex. The current check is a practical limit on K if N_coeffs_total is small.
    # Let's refine the check to ensure K is not excessively large for the given N_coeffs_total,
    # and that there are enough coefficients for at least one bit if K=1.
    if K > N_coeffs_total and M_total_bits > 0 : # If K is larger than all available coefficients
         # This is not necessarily an error if M_total_bits is small and K is reduced to N_coeffs_total for each bit.
         # However, it might indicate a misconfiguration.
         logging.warning(f"K ({K}) is larger than total available coefficients ({N_coeffs_total}). Effective K will be capped at {N_coeffs_total}.")
    if M_total_bits > N_coeffs_total and K == 1 : # If K=1 and still not enough coeffs
        raise EmbeddingError(f"Not enough coefficients ({N_coeffs_total}) in image to embed {M_total_bits} bits, even with K=1.")
    # The original check `M_total_bits > N_coeffs_total * K` was more about the number of modifications rather than distinct locations.
    # Let's keep a similar check for now, as it prevents K from being too small relative to M and N.
    # If we need to embed M_total_bits, and each takes K coefficients (potentially overlapping for different bits),
    # the critical factor is whether random.sample can pick K coefficients for each of M_total_bits.
    # The existing code uses min(K, N_coeffs_total) for idx_list size.
    # A simple check: if N_coeffs_total < some_minimum_for_K (e.g. K itself, or 1), then problem.
    if N_coeffs_total < 1 : # Already handled by N_coeffs_total == 0
        raise EmbeddingError("Coefficient pool is effectively empty.")

    # Derive a seed for the local PRNG from the main passphrase (key)
    # Using a different salt for this seed derivation is crucial.
    rng_seed_bytes: bytes = derive_aes_key(key, PBKDF2_SALT_RNG_SEED, key_length=PRNG_SEED_DERIVATION_LENGTH_BYTES) # key is already str
    rng_seed_int: int = int.from_bytes(rng_seed_bytes, BYTE_ORDER_BIG)
    rng: random.Random = random.Random(rng_seed_int) # Local PRNG instance

    for i in tqdm(range(M_total_bits), desc="Embedding bits"):
        # The original algorithm re-seeds for each bit using the main key + bit index.
        # This ensures that the coefficient selection for each bit is independent and deterministic based on the main key and bit position.
        # We apply this to our local rng instance.
        current_bit_seed = rng_seed_int + i
        rng.seed(current_bit_seed)

        # Ensure K does not exceed available coefficients
        actual_K = min(K, N_coeffs_total)
        if actual_K == 0 and N_coeffs_total > 0 : # Should not happen if N_coeffs_total > 0
            actual_K = 1 # Embed in at least one coefficient if available
        elif N_coeffs_total == 0: # This case is already checked above and raises EmbeddingError
             raise EmbeddingError("Coefficient pool is empty. Cannot embed.")

        idx_list = rng.sample(range(N_coeffs_total), actual_K)
        p_pattern = [rng.choice(BIPOLAR_MAPPING_VALUES) for _ in range(actual_K)]

        m_i_bipolar = BIPOLAR_CONVERSION_MULTIPLIER * encoded_bits_total[i] - BIPOLAR_CONVERSION_SUBTRACTOR # Convert bit 0/1 to -1/1

        current_adaptive_alpha_sum = 0 # For logging average alpha
        for k_idx, pool_idx in enumerate(idx_list):
            bi, bj, c_idx, u, v = pool[pool_idx]
            variance = block_variances[c_idx][bi * num_blocks_j + bj]
            # Ensure adaptive_alpha calculation is robust
            adaptive_alpha = alpha * (1 + variance / max(variance, VARIANCE_SCALING_FACTOR)) # Avoid division by zero if variance is 0
            adaptive_alpha = min(adaptive_alpha, MAX_ALPHA)
            current_adaptive_alpha_sum += adaptive_alpha

            # dct_blocks is now dct_blocks from compute_dct_blocks helper
            dct_blocks[c_idx][bi * num_blocks_j + bj][u, v] += adaptive_alpha * m_i_bipolar * p_pattern[k_idx]

        avg_adaptive_alpha: float = current_adaptive_alpha_sum / actual_K if actual_K > 0 else alpha # Type hint added
        if i % LOG_INTERVAL_EMBED_BITS == 0: # Log periodically
            logging.debug(f"Bit {i} embedded with avg_adaptive_alpha={avg_adaptive_alpha:.4f}, K_actual={actual_K}") # Changed to DEBUG

    # Reconstruct image using the helper function
    final_watermarked_image: np.ndarray = reconstruct_image_from_dct_blocks(dct_blocks, h_pad, w_pad, h_orig, w_orig)

    cv2.imwrite(str(output_path), final_watermarked_image)
    logging.info(f"Image successfully watermarked and saved to '{output_path}'")

def extract(image_path_str: str, key: str, K: int) -> str:
    """
    Extracts a secret message from a watermarked color image.
    Uses DCT, Hamming decoding, and AES decryption. Includes adaptive retry logic.

    Args:
        image_path_str: Path to the input watermarked image (PNG or JPEG).
        key: Passphrase (string) used during embedding for AES key derivation and PRNG seeding.
        K: Initial number of DCT coefficients to query per bit.

    Returns:
        The extracted secret message as a string.

    Raises:
        ConfigurationError: For invalid input parameters.
        ImageProcessingError: For issues with image loading or format.
        ExtractionError: For errors during the extraction process (e.g., failure to decode,
                         CRC mismatch after max attempts).
        CryptoError: For errors during cryptographic operations.
    """
    # --- Input Validation ---
    if K <= 0:
        raise ConfigurationError(f"K value {K} must be positive.")

    # Load, validate, and pad the image
    img_padded, _, _, h_pad, w_pad = load_and_pad_image(image_path_str) # Don't need h_orig, w_orig for extraction logic directly

    # Split into channels
    b, g, r = cv2.split(img_padded)
    channels_data = [b, g, r]

    # Compute DCT blocks (variances not needed for extraction)
    num_blocks_i: int = h_pad // IMAGE_BLOCK_SIZE # Type hint added
    num_blocks_j: int = w_pad // IMAGE_BLOCK_SIZE # Type hint added
    dct_blocks: List[List[np.ndarray]] # Type hint added
    dct_blocks, _ = compute_dct_blocks(channels_data, h_pad, w_pad, compute_variances=False)

    # Build coefficient pool
    coeff_pool: List[Tuple[int,int,int,int,int]] = build_coefficient_pool(num_blocks_i, num_blocks_j, IMAGE_CHANNELS)
    N_coeffs_total: int = len(coeff_pool)
    logging.info(f"Coefficient pool size: {N_coeffs_total}")

    if N_coeffs_total == 0:
        raise ImageProcessingError("Coefficient pool is empty. Cannot extract from this image (e.g. too small or no suitable AC coeffs).")

    # Derive a seed for the local PRNG from the main passphrase (key)
    rng_seed_bytes: bytes = derive_aes_key(key, PBKDF2_SALT_RNG_SEED, key_length=PRNG_SEED_DERIVATION_LENGTH_BYTES) # key is already str
    rng_seed_int: int = int.from_bytes(rng_seed_bytes, BYTE_ORDER_BIG)
    rng: random.Random = random.Random(rng_seed_int) # Local PRNG instance

    # --- Adaptive Extraction Attempt Loop ---
    # Start with initial K, then adjust if decoding fails
    current_K_extract = K
    # Alpha is not directly used in extraction formula, but K adjustment serves similar purpose of robustness

    L_extracted = INITIAL_INVALID_LENGTH # Initialize to invalid value

    for attempt in range(MAX_EXTRACTION_ATTEMPTS):
        logging.info(f"Extraction Attempt {attempt + 1}/{MAX_EXTRACTION_ATTEMPTS} with K_extract={current_K_extract}")

        len_encoded_extracted_bits = []
        for i in tqdm(range(ENCODED_LENGTH_BITS), desc=f"Extracting length (attempt {attempt+1}, K={current_K_extract})"):
            current_bit_seed = rng_seed_int + i
            rng.seed(current_bit_seed)

            actual_K_ext = min(current_K_extract, N_coeffs_total)
            if actual_K_ext == 0 : actual_K_ext = 1 # Ensure at least one coeff if pool not empty

            idx_list = rng.sample(range(N_coeffs_total), actual_K_ext)
            p_pattern = [rng.choice(BIPOLAR_MAPPING_VALUES) for _ in range(actual_K_ext)]

            sum_corr = 0
            for k_idx, pool_idx in enumerate(idx_list):
                bi, bj, c_idx, u, v = pool[pool_idx]
                sum_corr += dct_blocks[c_idx][bi * num_blocks_j + bj][u, v] * p_pattern[k_idx]

            # Average correlation. If actual_K_ext is 0, c_val is 0.
            c_val = sum_corr / actual_K_ext if actual_K_ext > 0 else 0
            extracted_bit = PADDING_BIT if c_val <= 0 else 1 # Use PADDING_BIT for 0
            len_encoded_extracted_bits.append(extracted_bit)
            if i % LOG_INTERVAL_EXTRACT_LENGTH_BITS == 0: # Log periodically
                 logging.debug(f"Length bit {i}, sum_corr={sum_corr:.4f}, c_val={c_val:.4f}, bit={extracted_bit}, K_actual={actual_K_ext}")

        len_bits_decoded_payload = []
        valid_hamming_chunks = True
        for j in range(0, ENCODED_LENGTH_BITS, HAMMING_CODEWORD_BITS):
            chunk = len_encoded_extracted_bits[j:j+HAMMING_CODEWORD_BITS]
            try:
                decoded_chunk = hamming_decode(chunk)
                len_bits_decoded_payload.extend(decoded_chunk)
            except AssertionError: # Should not happen if chunk length is correct
                logging.warning(f"Hamming decode failed for length chunk: {chunk}")
                valid_hamming_chunks = False
                break

        if not valid_hamming_chunks:
            logging.info(f"Attempt {attempt+1} failed: Error decoding Hamming chunks for length. Adjusting K_extract.") # INFO for attempt failure
            current_K_extract = min(current_K_extract * K_RETRY_MULTIPLIER, N_coeffs_total) # Increase K for next attempt
            if current_K_extract == K and K >= N_coeffs_total : # K is already maxed out
                logging.info(f"K_extract is already at maximum effective value ({K}). No further increase possible for K_extract.")
            continue # Try next attempt

        L_extracted = int(''.join(map(str, len_bits_decoded_payload[:LENGTH_FIELD_BITS])), BINARY_BASE)
        logging.info(f"Extracted raw length (decoded): {L_extracted} bits")

        # Validate L: must be positive, within representable range, and a multiple of EXPECTED_L_MULTIPLE_BITS
        if L_extracted > 0 and L_extracted <= MAX_L_VALUE and L_extracted % EXPECTED_L_MULTIPLE_BITS == 0:
            logging.info(f"Valid message bit length L={L_extracted} extracted after {attempt+1} attempts.")
            break # Valid L found, proceed to message extraction
        else:
            logging.info(f"Attempt {attempt+1}: Extracted length L={L_extracted} is invalid or unsuitable. " # INFO for attempt failure
                            f"Constraints: 0 < L <= {MAX_L_VALUE}, L % {EXPECTED_L_MULTIPLE_BITS} == 0. Adjusting K_extract.")
            current_K_extract = min(current_K_extract * K_RETRY_MULTIPLIER, N_coeffs_total) # Increase K
            L_extracted = INITIAL_INVALID_LENGTH # Reset L_extracted to invalid for next attempt or if all attempts fail
            if current_K_extract == K and K >= N_coeffs_total and attempt < MAX_EXTRACTION_ATTEMPTS -1: # K already maxed
                 logging.info(f"K_extract already at max effective value ({K}). Retrying with same K_extract for length.")
            elif attempt == MAX_EXTRACTION_ATTEMPTS -1: # Max attempts reached
                 logging.error(f"Max attempts ({MAX_EXTRACTION_ATTEMPTS}) reached for length extraction. Failed to get valid L.")


    if L_extracted <= 0: # Check if L extraction ultimately failed
        raise ExtractionError("Failed to extract valid message length (L) after maximum attempts.")

    # --- Extract Message Bits ---
    # Number of data bits in the message payload is L.
    # This L already includes IV and AES ciphertext.
    # Number of Hamming data chunks for the message part
    num_msg_data_chunks = (L_extracted + HAMMING_DATA_BITS - 1) // HAMMING_DATA_BITS
    # Total number of encoded bits for the message part
    M_msg_encoded_bits = num_msg_data_chunks * HAMMING_CODEWORD_BITS

    msg_extracted_text = "" # Initialize to empty string

    # Reset K for message extraction attempts (or continue with adjusted K from length extraction)
    # For simplicity, let's reset K, but could also inherit.
    current_K_extract_msg = K

    for attempt_msg in range(MAX_EXTRACTION_ATTEMPTS):
        logging.info(f"Message Extraction Attempt {attempt_msg + 1}/{MAX_EXTRACTION_ATTEMPTS} with K_extract_msg={current_K_extract_msg}")
        msg_encoded_extracted_bits = []
        # Start extracting from bit index after encoded length bits
        bit_offset = ENCODED_LENGTH_BITS
        for i in tqdm(range(M_msg_encoded_bits), desc=f"Extracting message (attempt {attempt_msg+1}, K={current_K_extract_msg})"):
            global_bit_index = bit_offset + i
            current_bit_seed = rng_seed_int + global_bit_index # Use the global bit index for reseeding
            rng.seed(current_bit_seed)

            actual_K_ext_msg = min(current_K_extract_msg, N_coeffs_total)
            if actual_K_ext_msg == 0: actual_K_ext_msg = 1

            idx_list = rng.sample(range(N_coeffs_total), actual_K_ext_msg)
            p_pattern = [rng.choice(BIPOLAR_MAPPING_VALUES) for _ in range(actual_K_ext_msg)]

            sum_corr = 0
            for k_idx, pool_idx in enumerate(idx_list):
                bi, bj, c_idx, u, v = pool[pool_idx]
                sum_corr += dct_blocks[c_idx][bi * num_blocks_j + bj][u, v] * p_pattern[k_idx]

            c_val = sum_corr / actual_K_ext_msg if actual_K_ext_msg > 0 else 0
            extracted_bit = PADDING_BIT if c_val <= 0 else 1 # Use PADDING_BIT for 0
            msg_encoded_extracted_bits.append(extracted_bit)
            if i % LOG_INTERVAL_EXTRACT_MESSAGE_BITS == 0: # Log periodically
                logging.debug(f"Message bit {i} (global {global_bit_index}), sum_corr={sum_corr:.4f}, c_val={c_val:.4f}, bit={extracted_bit}, K_actual={actual_K_ext_msg}")

        msg_bits_decoded_payload = []
        valid_msg_hamming_chunks = True
        for j in range(0, M_msg_encoded_bits, HAMMING_CODEWORD_BITS):
            chunk = msg_encoded_extracted_bits[j:j+HAMMING_CODEWORD_BITS]
            # Ensure chunk is full before attempting to decode
            if len(chunk) < HAMMING_CODEWORD_BITS:
                logging.warning(f"Skipping incomplete Hamming chunk for message: {chunk}")
                valid_msg_hamming_chunks = False # Or handle padding if expected
                break
            try:
                decoded_chunk = hamming_decode(chunk)
                msg_bits_decoded_payload.extend(decoded_chunk)
            except AssertionError:
                logging.warning(f"Hamming decode failed for message chunk: {chunk}")
                valid_msg_hamming_chunks = False
                break

        if not valid_msg_hamming_chunks:
            logging.info(f"Message Attempt {attempt_msg+1} failed: Error decoding Hamming chunks for message. Adjusting K_extract_msg.") # INFO
            current_K_extract_msg = min(current_K_extract_msg * K_RETRY_MULTIPLIER, N_coeffs_total)
            continue

        # Trim to actual length L_extracted (these are bits for IV + ciphertext)
        final_msg_bits_payload: List[int] = msg_bits_decoded_payload[:L_extracted]
        if len(final_msg_bits_payload) != L_extracted:
            logging.info(f"Message Attempt {attempt_msg+1}: Decoded message bit length {len(final_msg_bits_payload)} does not match expected L={L_extracted}. Adjusting K_extract_msg.") #INFO
            current_K_extract_msg = min(current_K_extract_msg * K_RETRY_MULTIPLIER, N_coeffs_total)
            continue

        # Convert bits to bytes (IV + Ciphertext)
        extracted_bytes_with_iv = bytes([int(''.join(map(str, final_msg_bits_payload[i:i+BITS_IN_BYTE])), BINARY_BASE) for i in range(0, L_extracted, BITS_IN_BYTE)])

        try:
            decrypted_bytes_with_crc: bytes = aes_decrypt(extracted_bytes_with_iv, key) # key is already str

            if len(decrypted_bytes_with_crc) < CRC_SIZE_BYTES:
                logging.info(f"Message Attempt {attempt_msg+1}: Decrypted data too short for CRC check. Adjusting K_extract_msg.") #INFO
                current_K_extract_msg = min(current_K_extract_msg * K_RETRY_MULTIPLIER, N_coeffs_total)
                continue

            received_crc_bytes: bytes = decrypted_bytes_with_crc[-CRC_SIZE_BYTES:]
            message_payload_bytes: bytes = decrypted_bytes_with_crc[:-CRC_SIZE_BYTES]

            received_crc: int = int.from_bytes(received_crc_bytes, BYTE_ORDER_BIG)
            computed_crc: int = compute_crc32(message_payload_bytes)

            if received_crc == computed_crc:
                logging.info("CRC verified successfully.")
                try:
                    msg_extracted_text = message_payload_bytes.decode(DEFAULT_ENCODING)
                    # Validate extracted characters (allow printable ASCII and common whitespace)
                    if all(PRINTABLE_ASCII_MIN <= ord(char) < PRINTABLE_ASCII_MAX or char in ALLOWED_WHITESPACE_CHARS for char in msg_extracted_text):
                        logging.info(f"Message decoded successfully after {attempt_msg+1} attempts: '{msg_extracted_text[:LOG_PREVIEW_LENGTH]}...'")
                        break # Success
                    else:
                        logging.warning(f"Message Attempt {attempt_msg+1}: CRC matched, but decoded message contains unexpected (non-ASCII/non-whitespace) characters. Treating as soft failure.")
                except UnicodeDecodeError:
                    logging.warning(f"Message Attempt {attempt_msg+1}: CRC matched, but {DEFAULT_ENCODING} decoding failed. Adjusting K_extract_msg.")
            else:
                logging.info(f"Message Attempt {attempt_msg+1}: CRC mismatch. Received: {received_crc}, Computed: {computed_crc}. Adjusting K_extract_msg.") #INFO

        except CryptoError as e: # Catch specific AES decryption errors (e.g., padding)
            logging.warning(f"Message Attempt {attempt_msg+1}: CryptoError during decryption/CRC check: {str(e)}. Adjusting K_extract_msg.")
        except Exception as e: # Catch other unexpected errors during this critical section
            logging.error(f"Message Attempt {attempt_msg+1}: Unexpected error during decryption/CRC check: {str(e)}", exc_info=True)
            # This will be treated as a failed attempt, and K will be adjusted.

        # If loop continues, it means this attempt failed (CRC mismatch, decode error, etc.)
        current_K_extract_msg = min(current_K_extract_msg * K_RETRY_MULTIPLIER, N_coeffs_total)
        if current_K_extract_msg == K and K >= N_coeffs_total and attempt_msg < MAX_EXTRACTION_ATTEMPTS -1 :
            logging.info(f"K_extract_msg already at max effective value ({K}). Retrying with same K_extract_msg.")
        elif attempt_msg == MAX_EXTRACTION_ATTEMPTS -1:
            logging.error(f"Max attempts ({MAX_EXTRACTION_ATTEMPTS}) reached for message extraction. Failed to verify message.")


    if not msg_extracted_text: # Check if message extraction ultimately failed
        raise ExtractionError("Failed to extract and verify message after maximum attempts.")

    return msg_extracted_text

# --- Main Execution Block ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Advanced DCT-based Spread Spectrum Watermarking Tool with Error Correction and AES Encryption.",
        formatter_class=argparse.RawTextHelpFormatter # For better help text formatting
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO)."
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands: 'embed' or 'extract'")

    # Embed command
    embed_parser = subparsers.add_parser(CMD_EMBED, help="Embed a message into a color image.")
    embed_parser.add_argument("--image", required=True, type=str, help="Input color image path (e.g., input.png).")
    embed_parser.add_argument("--message", required=True, type=str, help=f"Message to embed (max {MAX_MESSAGE_CHARS} printable ASCII-like chars).")
    embed_parser.add_argument("--output", required=True, type=str, help="Output watermarked color image path (e.g., output.png).")
    embed_parser.add_argument("--key", required=True, type=str, help="Passphrase for encryption and PRNG seeding.") # Changed to str
    embed_parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help=f"Initial embedding strength (default: {DEFAULT_ALPHA}, max: {MAX_ALPHA}). Should be > 0.")
    embed_parser.add_argument("--K", type=int, default=DEFAULT_K_COEFFS, help=f"Number of DCT coefficients to modify per bit (default: {DEFAULT_K_COEFFS}). Should be > 0.")

    # Extract command
    extract_parser = subparsers.add_parser(CMD_EXTRACT, help="Extract a message from a color image.")
    extract_parser.add_argument("--image", required=True, type=str, help="Input watermarked color image path (e.g., output.png).")
    extract_parser.add_argument("--key", required=True, type=str, help="Passphrase used during embedding.") # Changed to str
    extract_parser.add_argument("--K", type=int, default=DEFAULT_K_COEFFS, help=f"Initial number of DCT coefficients to query per bit (default: {DEFAULT_K_COEFFS}). Should be > 0.")

    args = parser.parse_args()

    # Basic validation for K and alpha at CLI level as well
    if hasattr(args, 'K') and args.K <= 0:
        parser.error("Argument --K: must be a positive integer.")
    if hasattr(args, 'alpha') and args.alpha <= 0: # alpha can be > MAX_ALPHA, will be capped in embed()
        parser.error("Argument --alpha: must be a positive float.")

    # Configure logging level based on CLI argument
    # Initial basicConfig is done at the top of the script with default INFO level to a file.
    # Here, we update the level of the root logger and its handlers.
    numeric_log_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_log_level, int):
        # This should not happen due to argparse 'choices'
        logging.error(f"Invalid log level: {args.log_level}. Defaulting to INFO.")
        numeric_log_level = logging.INFO

    logging.getLogger().setLevel(numeric_log_level)
    # Update levels for all handlers attached to the root logger
    for handler in logging.getLogger().handlers:
        handler.setLevel(numeric_log_level)

    logging.info(f"Logging level set to {args.log_level.upper()}")


    try:
        if args.command == CMD_EMBED:
            # Message character validation (basic check, can be expanded)
            if not all(PRINTABLE_ASCII_MIN <= ord(char) < PRINTABLE_ASCII_MAX or char in ALLOWED_WHITESPACE_CHARS for char in args.message):
                 print(f"Warning: Message contains characters outside the standard printable ASCII range (ASCII {PRINTABLE_ASCII_MIN}-{PRINTABLE_ASCII_MAX-1}) or allowed whitespace ({ALLOWED_WHITESPACE_CHARS}). This might affect recovery or display.", file=sys.stderr)

            embed(args.image, args.message, args.output, args.key, args.alpha, args.K)
            print(f"Message embedded successfully into '{args.output}'")
        elif args.command == CMD_EXTRACT:
            message: str = extract(args.image, args.key, args.K)
            print(f"Extracted message: {message}")

    except WatermarkingError as e: # Catch specific custom errors first
        logging.error(f"Watermarking operation failed: {str(e)}")
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(ERROR_EXIT_CODE)
    except Exception as e: # Catch any other unexpected errors
        logging.critical(f"An unexpected critical error occurred: {str(e)}", exc_info=True) # Log with stack trace
        print(f"An unexpected critical error occurred. Please check the log file: '{LOG_FILE_NAME}' for details.", file=sys.stderr)
        sys.exit(ERROR_EXIT_CODE)