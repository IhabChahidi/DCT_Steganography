import unittest
import os # For os.urandom if needed directly in tests, though aes_encrypt handles it
import zlib # For test_crc32_computation if a known value is used.

# Assuming functions are importable from main.py
# If main.py is structured as a script, these might need adjustment,
# or main.py might need to be refactored to expose these for testing.
from main import (
    hamming_encode,
    hamming_decode,
    aes_encrypt,
    aes_decrypt,
    compute_crc32,
    # Constants that might be needed for tests
    HAMMING_DATA_BITS,
    HAMMING_CODEWORD_BITS,
    IV_SIZE_BYTES
)

class TestHammingCodes(unittest.TestCase):
    def test_hamming_encode_decode_no_error(self):
        """Test Hamming encoding and decoding with no errors."""
        original_data = [1, 0, 1, 0]
        encoded_data = hamming_encode(original_data)
        self.assertEqual(len(encoded_data), HAMMING_CODEWORD_BITS)
        decoded_data = hamming_decode(encoded_data)
        self.assertEqual(decoded_data, original_data)

        original_data_2 = [0, 1, 1, 1]
        encoded_data_2 = hamming_encode(original_data_2)
        decoded_data_2 = hamming_decode(encoded_data_2)
        self.assertEqual(decoded_data_2, original_data_2)

    def test_hamming_decode_single_bit_error(self):
        """Test Hamming decoding with a single bit error correction."""
        original_data = [1, 1, 0, 1]
        encoded_data = hamming_encode(original_data)

        # Introduce a single bit error
        error_position = 3
        encoded_data_with_error = list(encoded_data) # Make a mutable copy
        encoded_data_with_error[error_position] = 1 - encoded_data_with_error[error_position] # Flip the bit

        decoded_data = hamming_decode(encoded_data_with_error)
        self.assertEqual(decoded_data, original_data, "Should correct single bit error")

        # Test another case
        original_data_2 = [0, 0, 1, 0]
        encoded_data_2 = hamming_encode(original_data_2)
        error_position_2 = 0
        encoded_data_with_error_2 = list(encoded_data_2)
        encoded_data_with_error_2[error_position_2] = 1 - encoded_data_with_error_2[error_position_2]
        decoded_data_2 = hamming_decode(encoded_data_with_error_2)
        self.assertEqual(decoded_data_2, original_data_2, "Should correct single bit error at first position")


    def test_hamming_encode_invalid_input(self):
        """Test Hamming encode with invalid input length."""
        with self.assertRaises(AssertionError): # As per current main.py
            hamming_encode([1, 0, 1]) # Too short
        with self.assertRaises(AssertionError):
            hamming_encode([1, 0, 1, 0, 1]) # Too long

    def test_hamming_decode_invalid_input(self):
        """Test Hamming decode with invalid input length."""
        with self.assertRaises(AssertionError): # As per current main.py
            hamming_decode([1, 0, 1, 0, 1, 0]) # Too short
        with self.assertRaises(AssertionError):
            hamming_decode([1, 0, 1, 0, 1, 0, 1, 1]) # Too long

class TestAESEncryption(unittest.TestCase):
    def test_aes_encrypt_decrypt_basic(self):
        """Test AES encryption and decryption for a basic message."""
        key = 12345
        original_message = b"This is a secret message for AES."

        encrypted_data = aes_encrypt(original_message, key)
        self.assertIsNotNone(encrypted_data)
        self.assertNotEqual(encrypted_data, original_message)

        # Check that IV is prepended (output is longer than input by IV_SIZE_BYTES + padding)
        # AES pads to 16 byte blocks. If message is 33 bytes, it pads to 48.
        # So len(encrypted) = IV_SIZE_BYTES (16) + 48 = 64
        # If original_message was 16 bytes, padded would be 32. len(encrypted) = 16 + 32 = 48
        expected_min_len = IV_SIZE_BYTES + len(original_message)
        self.assertTrue(len(encrypted_data) >= expected_min_len)
        self.assertTrue((len(encrypted_data) - IV_SIZE_BYTES) % 16 == 0) # Padded part is multiple of block size

        decrypted_message = aes_decrypt(encrypted_data, key)
        self.assertEqual(decrypted_message, original_message)

    def test_aes_encrypt_decrypt_empty_message(self):
        """Test AES encryption and decryption for an empty message."""
        key = 98765
        original_message = b"" # Empty message
        encrypted_data = aes_encrypt(original_message, key)
        self.assertIsNotNone(encrypted_data)
        # IV (16 bytes) + 1 block of padding (16 bytes) = 32 bytes
        self.assertEqual(len(encrypted_data), IV_SIZE_BYTES + 16)

        decrypted_message = aes_decrypt(encrypted_data, key)
        self.assertEqual(decrypted_message, original_message)

    def test_aes_decrypt_tampered_data(self):
        """Test AES decryption with tampered ciphertext."""
        key = 67890
        original_message = b"Another secret to test tampering."
        encrypted_data = aes_encrypt(original_message, key)

        # Tamper the ciphertext (excluding the IV)
        tampered_encrypted_data = bytearray(encrypted_data)
        if len(tampered_encrypted_data) > IV_SIZE_BYTES + 1: # Ensure there is ciphertext to tamper
            tampered_encrypted_data[IV_SIZE_BYTES + 0] = tampered_encrypted_data[IV_SIZE_BYTES + 0] ^ 0xFF # Flip first byte of ciphertext
        else: # This case should not happen with non-empty messages due to padding
             self.fail("Encrypted data too short to tamper for this test.")

        with self.assertRaisesRegex(ValueError, "AES decryption failed, possibly due to incorrect key or corrupted data."):
            aes_decrypt(bytes(tampered_encrypted_data), key)

    def test_aes_decrypt_wrong_key(self):
        """Test AES decryption with an incorrect key."""
        key1 = 11111
        key2 = 22222
        original_message = b"Message for key test."
        encrypted_data = aes_encrypt(original_message, key1)

        with self.assertRaisesRegex(ValueError, "AES decryption failed, possibly due to incorrect key or corrupted data."):
            aes_decrypt(encrypted_data, key2)

    def test_aes_decrypt_invalid_padding(self):
        """Test AES decryption with data that would result in invalid padding."""
        key = 33333
        # Construct data that is not correctly padded after IV stripping
        # IV (16 bytes) + non-padded or incorrectly padded data (e.g. 15 bytes)
        iv = os.urandom(IV_SIZE_BYTES)
        bad_ciphertext = os.urandom(15) # Not a multiple of AES block size
        encrypted_data_bad_padding = iv + bad_ciphertext

        with self.assertRaisesRegex(ValueError, "AES decryption failed, possibly due to incorrect key or corrupted data."):
            aes_decrypt(encrypted_data_bad_padding, key)

        # Another case: last byte indicates padding length X, but previous X-1 bytes are not all X
        iv2 = os.urandom(IV_SIZE_BYTES)
        # Ciphertext: [..., byte_val, byte_val, ..., N] where N is pad_length
        # If N=3, last three bytes are 0x03, 0x03, 0x03.
        # Create something like: [..., 0x01, 0x02, 0x03]
        ciphertext_manual_bad_padding = os.urandom(13) + b'\x01\x02\x03' # 16 bytes total
        encrypted_data_manual_bad_padding = iv2 + ciphertext_manual_bad_padding

        # This specific type of bad padding is caught by "Invalid PKCS7 padding bytes found."
        # which is then re-raised as the more generic "AES decryption failed..."
        with self.assertRaisesRegex(ValueError, "AES decryption failed, possibly due to incorrect key or corrupted data."):
            aes_decrypt(encrypted_data_manual_bad_padding, key)

class TestCRC32(unittest.TestCase):
    def test_crc32_computation(self):
        """Test CRC32 computation with a known value."""
        # Known CRC32 value for "hello world" (ASCII)
        # Can be verified with online tools or other libraries.
        # Example: zlib.crc32(b"hello world")
        data = b"hello world"
        expected_crc = 2983298053 # zlib.crc32(b"hello world") & 0xFFFFFFFF

        # Our function also does & 0xFFFFFFFF
        self.assertEqual(compute_crc32(data), expected_crc)

    def test_crc32_consistency(self):
        """Test that CRC32 is consistent for the same input."""
        data = b"some random data for consistency check"
        crc1 = compute_crc32(data)
        crc2 = compute_crc32(data)
        self.assertEqual(crc1, crc2)

    def test_crc32_different_for_different_data(self):
        """Test that CRC32 is different for different inputs."""
        data1 = b"data one"
        data2 = b"data two"
        self.assertNotEqual(compute_crc32(data1), compute_crc32(data2))

    def test_crc32_empty_data(self):
        """Test CRC32 computation for empty input."""
        # zlib.crc32(b"") is 0
        self.assertEqual(compute_crc32(b""), 0)

# Need to import more for integration tests
import numpy as np
import cv2 # OpenCV for image operations like imwrite
import tempfile # For temporary file creation

# Import functions to be tested
from main import embed, extract, BLOCK_SIZE # BLOCK_SIZE is needed for image dim

class TestEmbedExtract(unittest.TestCase):
    def setUp(self):
        """Set up common variables for embed/extract tests."""
        self.sample_message = "Test Msg 123! @#$"
        self.sample_key = 777
        self.sample_alpha = 0.5
        self.sample_K = 10 # Smaller K for faster test, but ensure it's > 0

        # Create a simple, small image (multiples of BLOCK_SIZE)
        # Ensure dimensions are sufficient for K value to not exhaust pool too quickly.
        # For K=10, pool size needs to be at least 10.
        # Pool size = (H/B-1)*(W/B-1)*3*(B-1)*(B-1) -> simplified: (num_blocks_i-1)*(num_blocks_j-1)*3*49
        # Smallest image: 2*BLOCK_SIZE x 2*BLOCK_SIZE to have (1*1*3*49) = 147 pool size if BLOCK_SIZE=8
        height = 2 * BLOCK_SIZE
        width = 2 * BLOCK_SIZE
        self.original_image_array = np.random.randint(0, 256, size=(height, width, 3), dtype=np.uint8)

        # Temporary files for images
        # Suffix is important for cv2.imwrite to know the format
        self.original_image_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        self.watermarked_image_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)

        self.original_image_path = self.original_image_file.name
        self.watermarked_image_path = self.watermarked_image_file.name

        # Save the dummy image
        cv2.imwrite(self.original_image_path, self.original_image_array)
        self.original_image_file.close() # Close it so embed can open it if needed
        self.watermarked_image_file.close()


    def tearDown(self):
        """Clean up temporary files."""
        os.remove(self.original_image_path)
        if os.path.exists(self.watermarked_image_path): # Watermarked might not be created if embed fails early
            os.remove(self.watermarked_image_path)

    def test_embed_extract_successful(self):
        """Test the full embed and extract process successfully."""
        try:
            embed(
                image_path=self.original_image_path,
                message=self.sample_message,
                output_path=self.watermarked_image_path,
                key=self.sample_key,
                alpha=self.sample_alpha,
                K=self.sample_K
            )

            # Ensure watermarked file was created and has content
            self.assertTrue(os.path.exists(self.watermarked_image_path))
            self.assertTrue(os.path.getsize(self.watermarked_image_path) > 0)

            extracted_message = extract(
                image_path=self.watermarked_image_path,
                key=self.sample_key,
                K=self.sample_K
            )
            self.assertEqual(extracted_message, self.sample_message)
        except Exception as e:
            # Provide more context if test fails
            self.fail(f"test_embed_extract_successful failed with {type(e).__name__}: {e}")

    def test_extract_from_unwatermarked_image(self):
        """Test extracting from an unwatermarked image, expecting a ValueError."""
        # Attempt to extract from the original image which has no watermark
        with self.assertRaisesRegex(ValueError, "Failed to extract valid message length L after maximum attempts."):
            extract(
                image_path=self.original_image_path, # Use the original, non-watermarked image
                key=self.sample_key,
                K=self.sample_K
            )
            # The error message "Failed to extract valid message length L after maximum attempts."
            # is what _extract_bits_from_dct raises. If other ValueErrors occur first (e.g. CRC, decode)
            # this regex might need adjustment, but length extraction is the earliest expected failure point.

if __name__ == '__main__':
    unittest.main()
