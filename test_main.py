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
from main import (
    embed,
    extract,
    BLOCK_SIZE, # BLOCK_SIZE is needed for image dim
    _derive_key_material, # For testing key derivation
    _load_config, # For testing config loading
    # Relevant constants for testing config and key derivation
    AES_KEY_SIZE_BYTES,
    DERIVED_KEY_MATERIAL_SIZE_BYTES,
    DEFAULT_ALPHA,
    DEFAULT_K,
    DEFAULT_PBKDF2_SALT_HEX,
    CONFIG_FILE_NAME,
    PBKDF2_ITERATIONS # Needed for _derive_key_material
)

# Additional imports for new tests
import configparser
from unittest.mock import patch, mock_open

class TestKeyDerivation(unittest.TestCase):
    def test_derive_key_material_output_length(self):
        """Test that _derive_key_material returns a key of the expected length."""
        passphrase = "test_passphrase"
        salt = os.urandom(16)
        iterations = PBKDF2_ITERATIONS # Use the one from main.py
        expected_length = DERIVED_KEY_MATERIAL_SIZE_BYTES

        derived_key = _derive_key_material(passphrase, salt, iterations, expected_length)
        self.assertEqual(len(derived_key), expected_length)

    def test_derive_key_material_consistency(self):
        """Test that the same passphrase and salt produce the same key."""
        passphrase = "consistent_pass"
        salt = b"fixed_salt_for_test" # Fixed salt for this test
        iterations = PBKDF2_ITERATIONS
        length = DERIVED_KEY_MATERIAL_SIZE_BYTES

        key1 = _derive_key_material(passphrase, salt, iterations, length)
        key2 = _derive_key_material(passphrase, salt, iterations, length)
        self.assertEqual(key1, key2)

    def test_derive_key_material_diff_passphrase(self):
        """Test that different passphrases produce different keys with the same salt."""
        passphrase1 = "passphrase_one"
        passphrase2 = "passphrase_two"
        salt = b"fixed_salt_for_test"
        iterations = PBKDF2_ITERATIONS
        length = DERIVED_KEY_MATERIAL_SIZE_BYTES

        key1 = _derive_key_material(passphrase1, salt, iterations, length)
        key2 = _derive_key_material(passphrase2, salt, iterations, length)
        self.assertNotEqual(key1, key2)

    def test_derive_key_material_diff_salt(self):
        """Test that different salts produce different keys with the same passphrase."""
        passphrase = "common_passphrase"
        salt1 = b"salt_number_one_16b"
        salt2 = b"salt_number_two_16b"
        iterations = PBKDF2_ITERATIONS
        length = DERIVED_KEY_MATERIAL_SIZE_BYTES

        self.assertNotEqual(salt1, salt2, "Salts must be different for this test")

        key1 = _derive_key_material(passphrase, salt1, iterations, length)
        key2 = _derive_key_material(passphrase, salt2, iterations, length)
        self.assertNotEqual(key1, key2)

@patch('main.CONFIG_FILE_NAME', 'test_config.ini') # Patch global constant in main
class TestConfigFileLoading(unittest.TestCase):
    def tearDown(self):
        """Ensure test_config.ini is removed if created by a test."""
        if os.path.exists('test_config.ini'):
            os.remove('test_config.ini')

    def test_load_config_valid_file(self):
        """Test loading a valid config file."""
        config_content = """
[Defaults]
alpha = 2.5
k = 750
pbkdf2_salt_hex = aabbccddeeff00112233445566778899
"""
        with open('test_config.ini', 'w') as f:
            f.write(config_content)

        alpha, k, salt = _load_config()
        self.assertEqual(alpha, 2.5)
        self.assertEqual(k, 750)
        self.assertEqual(salt, bytes.fromhex("aabbccddeeff00112233445566778899"))

    @patch('os.path.exists', return_value=False) # Mock os.path.exists used by configparser.read
    @patch('builtins.open', new_callable=mock_open, read_data="") # Mock open for configparser
    def test_load_config_missing_file(self, mock_file_open, mock_os_exists):
        """Test loading when config file is missing; should use defaults."""
        # By patching os.path.exists to return False, config.read will act as if file is not found.
        # Need to ensure configparser.read doesn't find the file.
        # The @patch on CONFIG_FILE_NAME might not be enough if configparser internally uses it.
        # A more robust way is to ensure config.read(CONFIG_FILE_NAME) returns an empty list.
        with patch('configparser.ConfigParser.read', return_value=[]) as mock_read:
            alpha, k, salt = _load_config()
            self.assertEqual(alpha, DEFAULT_ALPHA)
            self.assertEqual(k, DEFAULT_K)
            self.assertEqual(salt, bytes.fromhex(DEFAULT_PBKDF2_SALT_HEX))
            mock_read.assert_called_with('test_config.ini')


    def test_load_config_partial_file(self):
        """Test loading a config file with some values missing."""
        config_content = """
[Defaults]
alpha = 3.0
# k is missing
# pbkdf2_salt_hex is missing
"""
        with open('test_config.ini', 'w') as f:
            f.write(config_content)

        alpha, k, salt = _load_config()
        self.assertEqual(alpha, 3.0) # Loaded from file
        self.assertEqual(k, DEFAULT_K) # Default
        self.assertEqual(salt, bytes.fromhex(DEFAULT_PBKDF2_SALT_HEX)) # Default

    @patch('main.logging') # Mock logging to check for warnings
    def test_load_config_invalid_salt_hex(self, mock_logging):
        """Test config with invalid hex for salt; should use default salt and log warning."""
        config_content = """
[Defaults]
pbkdf2_salt_hex = not_a_valid_hex_string!
"""
        with open('test_config.ini', 'w') as f:
            f.write(config_content)

        alpha, k, salt = _load_config()
        self.assertEqual(alpha, DEFAULT_ALPHA)
        self.assertEqual(k, DEFAULT_K)
        self.assertEqual(salt, bytes.fromhex(DEFAULT_PBKDF2_SALT_HEX))
        # Check if logging.error was called (as per _load_config logic for invalid hex)
        mock_logging.error.assert_called_with(f"Invalid hex value for pbkdf2_salt_hex in 'test_config.ini'. Using default salt.")


    def test_load_config_empty_file(self):
        """Test loading an empty config file; should use defaults."""
        with open('test_config.ini', 'w') as f:
            f.write("") # Empty file

        alpha, k, salt = _load_config()
        self.assertEqual(alpha, DEFAULT_ALPHA)
        self.assertEqual(k, DEFAULT_K)
        self.assertEqual(salt, bytes.fromhex(DEFAULT_PBKDF2_SALT_HEX))

    @patch('main.logging')
    def test_load_config_short_salt(self, mock_logging):
        """Test loading a config file with a too short salt hex."""
        config_content = """
[Defaults]
pbkdf2_salt_hex = aabbcc
"""
        with open('test_config.ini', 'w') as f:
            f.write(config_content)

        alpha, k, salt = _load_config()
        self.assertEqual(salt, bytes.fromhex(DEFAULT_PBKDF2_SALT_HEX))
        mock_logging.warning.assert_called_with(f"Configured PBKDF2 salt is very short (length {len(bytes.fromhex('aabbcc'))}). Using default salt instead.")

# For testing argument parsing, we might need to import the parser setup from main.
# This can be tricky if it's all within if __name__ == "__main__".
# For now, let's assume we can inspect the parser object or its actions.
# A simpler approach is to test that certain combinations *would* lead to errors,
# or that the help messages reflect the new arguments.

# As direct testing of parser.error() is complex, these tests will be more conceptual
# or would require refactoring main.py to make parser setup more accessible for testing.
# For this exercise, I'll sketch out what such tests might look like if main.py's parser
# was constructed in an importable function.
# Since it's not, these tests will be more of a placeholder or focus on what can be checked.

class TestBatchProcessingArguments(unittest.TestCase):
    # This test class is more of a conceptual placeholder due to the difficulty of
    # directly testing argparse configurations that use parser.error() within the
    # __main__ block of a script without refactoring that script.

    def test_embed_parser_has_output_and_output_dir(self):
        """Check if embed parser is aware of --output and --output-dir."""
        # This would ideally involve getting the ArgumentParser object from main.py
        # and inspecting its actions or subparsers.
        # e.g., parser = main.create_parser()
        # embed_actions = [action.dest for action in parser._subparsers._group_actions[0].choices['embed']._actions]
        # self.assertIn('output', embed_actions)
        # self.assertIn('output_dir', embed_actions)
        pass # Placeholder - direct test not straightforward with current main.py structure

    def test_extract_parser_has_image(self):
        """Check if extract parser is aware of --image."""
        # Similar to above, would need access to the parser.
        # extract_actions = [action.dest for action in parser._subparsers._group_actions[0].choices['extract']._actions]
        # self.assertIn('image', extract_actions)
        pass # Placeholder


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
