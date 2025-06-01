import unittest
import random # For introducing errors
import os # For os.urandom
import zlib # For getting known CRC32 values
import logging # For suppressing debug messages from main.py if necessary, or for test logging

from main import (
    hamming_encode, hamming_decode,
    encode_data_with_hamming, decode_data_with_hamming,
    derive_key, aes_encrypt, aes_decrypt,
    compute_crc32
)

# Suppress logging from the main module during tests if it's too verbose
# logging.getLogger('main').setLevel(logging.CRITICAL) # Example if main.py had a logger named 'main'
# For the current main.py, logging is configured with basicConfig, so specific suppression is harder.
# We can disable all logging below a certain level for the duration of tests if needed.
# logging.disable(logging.INFO) # Disables INFO and DEBUG, WARNING and above will still show.

class TestMainUtils(unittest.TestCase):

    # --- Hamming Code Tests ---
    def test_hamming_single_chunk_no_error(self):
        original_data = [1, 0, 1, 0]
        encoded_data = hamming_encode(original_data)
        self.assertEqual(len(encoded_data), 7)
        decoded_data = hamming_decode(encoded_data)
        self.assertEqual(decoded_data, original_data)

    def test_hamming_single_chunk_all_zeros(self):
        original_data = [0, 0, 0, 0]
        encoded_data = hamming_encode(original_data)
        decoded_data = hamming_decode(encoded_data)
        self.assertEqual(decoded_data, original_data)

    def test_hamming_single_chunk_all_ones(self):
        original_data = [1, 1, 1, 1]
        encoded_data = hamming_encode(original_data)
        decoded_data = hamming_decode(encoded_data)
        self.assertEqual(decoded_data, original_data)

    def test_hamming_single_chunk_with_error_correction(self):
        original_data = [1, 0, 1, 0]
        encoded_data = hamming_encode(original_data)
        error_position = random.randint(0, 6)
        encoded_data_with_error = list(encoded_data)
        encoded_data_with_error[error_position] = 1 - encoded_data_with_error[error_position]
        decoded_data = hamming_decode(encoded_data_with_error)
        self.assertEqual(decoded_data, original_data, "Hamming decode should correct single bit errors")

    def test_encode_data_with_hamming_no_padding(self):
        original_sequence = [1, 0, 1, 0, 0, 1, 1, 0] # 8 bits
        encoded_sequence = encode_data_with_hamming(original_sequence)
        self.assertEqual(len(encoded_sequence), 14)
        decoded_sequence = decode_data_with_hamming(encoded_sequence)
        self.assertEqual(decoded_sequence, original_sequence)

    def test_encode_data_with_hamming_with_padding(self):
        original_sequence = [1, 0, 1, 0, 0, 1] # 6 bits
        expected_padded_sequence = original_sequence + [0, 0]
        encoded_sequence = encode_data_with_hamming(original_sequence)
        self.assertEqual(len(encoded_sequence), 14)
        decoded_sequence = decode_data_with_hamming(encoded_sequence)
        self.assertEqual(decoded_sequence, expected_padded_sequence)
        self.assertEqual(decoded_sequence[:len(original_sequence)], original_sequence)

    def test_hamming_sequence_empty_input(self):
        self.assertEqual(encode_data_with_hamming([]), [])
        self.assertEqual(decode_data_with_hamming([]), [])

    def test_hamming_sequence_with_error_correction(self):
        original_sequence = [1,0,1,0, 0,1,1,0, 1,1,0,0] # 12 bits
        encoded_sequence = encode_data_with_hamming(original_sequence)
        self.assertEqual(len(encoded_sequence), 21)
        encoded_sequence_with_errors = list(encoded_sequence)
        if len(encoded_sequence_with_errors) >= 7:
            error_pos1 = random.randint(0, 6)
            encoded_sequence_with_errors[error_pos1] = 1 - encoded_sequence_with_errors[error_pos1]
        if len(encoded_sequence_with_errors) >= 14:
            error_pos2 = random.randint(7, 13)
            encoded_sequence_with_errors[error_pos2] = 1 - encoded_sequence_with_errors[error_pos2]
        decoded_sequence = decode_data_with_hamming(encoded_sequence_with_errors)
        self.assertEqual(decoded_sequence, original_sequence)

    def test_decode_data_with_hamming_invalid_length(self):
        invalid_encoded_sequence = [0, 1] * 5 # 10 bits
        with self.assertRaises(ValueError):
            decode_data_with_hamming(invalid_encoded_sequence)

    def test_hamming_specific_error_case(self):
        original_data = [0,1,0,1]
        encoded = hamming_encode(original_data)
        corrupted_encoded = list(encoded)
        corrupted_encoded[0] = 1 - corrupted_encoded[0]
        decoded = hamming_decode(corrupted_encoded)
        self.assertEqual(decoded, original_data)

    # --- Key Derivation Tests ---
    def test_derive_key_length(self):
        password = "test_password"
        salt = os.urandom(16)
        key_len = 32
        key = derive_key(password, salt, key_len)
        self.assertEqual(len(key), key_len)

    def test_derive_key_consistency(self):
        password = "test_password"
        salt = os.urandom(16)
        key1 = derive_key(password, salt, 32)
        key2 = derive_key(password, salt, 32)
        self.assertEqual(key1, key2)

    def test_derive_key_diff_salt(self):
        password = "test_password"
        salt1 = os.urandom(16)
        salt2 = os.urandom(16)
        while salt1 == salt2:
            salt2 = os.urandom(16)
        key1 = derive_key(password, salt1, 32)
        key2 = derive_key(password, salt2, 32)
        self.assertNotEqual(key1, key2)

    def test_derive_key_diff_password(self):
        password_a = "test_password_A"
        password_b = "test_password_B"
        salt = os.urandom(16)
        key1 = derive_key(password_a, salt, 32)
        key2 = derive_key(password_b, salt, 32)
        self.assertNotEqual(key1, key2)

    # --- AES Encryption/Decryption Tests ---
    def test_aes_encrypt_decrypt_basic(self):
        password = "aes_password"
        original_message_str = "Hello, AES!"
        original_message_bytes = original_message_str.encode('utf-8')
        encrypted_data = aes_encrypt(original_message_bytes, password)
        decrypted_message_bytes = aes_decrypt(encrypted_data, password)
        self.assertEqual(decrypted_message_bytes, original_message_bytes)

    def test_aes_encrypt_decrypt_empty_message(self):
        password = "aes_password_empty"
        original_message_str = ""
        original_message_bytes = original_message_str.encode('utf-8')
        encrypted_data = aes_encrypt(original_message_bytes, password)
        decrypted_message_bytes = aes_decrypt(encrypted_data, password)
        self.assertEqual(decrypted_message_bytes, original_message_bytes)

    def test_aes_encrypt_decrypt_long_message(self):
        password = "aes_password_long"
        original_message_str = "This is a much longer message for AES." * 10
        original_message_bytes = original_message_str.encode('utf-8')
        encrypted_data = aes_encrypt(original_message_bytes, password)
        decrypted_message_bytes = aes_decrypt(encrypted_data, password)
        self.assertEqual(decrypted_message_bytes, original_message_bytes)

    def test_aes_decrypt_wrong_password(self):
        password_correct = "aes_password_correct"
        password_wrong = "aes_password_WRONG"
        original_message_bytes = b"Secret message."
        encrypted_data = aes_encrypt(original_message_bytes, password_correct)
        try:
            decrypted_message_bytes = aes_decrypt(encrypted_data, password_wrong)
            self.assertNotEqual(decrypted_message_bytes, original_message_bytes)
        except ValueError as e:
            self.assertTrue("padding" in str(e).lower() or "decryption" in str(e).lower())

    def test_aes_decrypt_tampered_ciphertext(self):
        password = "aes_password_tamper_ct"
        original_message_bytes = b"Another secret."
        encrypted_data = aes_encrypt(original_message_bytes, password)
        tampered_encrypted_data = bytearray(encrypted_data)
        if len(tampered_encrypted_data) > 32:
            tamper_idx = random.randint(32, len(tampered_encrypted_data) - 1)
            tampered_encrypted_data[tamper_idx] = (tampered_encrypted_data[tamper_idx] + 1) % 256
        else:
            self.fail("Encrypted data too short to tamper ciphertext part.")
        try:
            decrypted_message_bytes = aes_decrypt(bytes(tampered_encrypted_data), password)
            self.assertNotEqual(decrypted_message_bytes, original_message_bytes)
        except ValueError as e:
            self.assertTrue("padding" in str(e).lower() or "decryption" in str(e).lower())

    def test_aes_decrypt_tampered_salt(self):
        password = "aes_password_tamper_salt"
        original_message_bytes = b"Salted secret."
        encrypted_data = aes_encrypt(original_message_bytes, password)
        tampered_encrypted_data = bytearray(encrypted_data)
        if len(tampered_encrypted_data) >= 16:
            tamper_idx = random.randint(0, 15)
            tampered_encrypted_data[tamper_idx] = (tampered_encrypted_data[tamper_idx] + 1) % 256
        else:
            self.fail("Encrypted data too short to tamper salt.")
        try:
            decrypted_message_bytes = aes_decrypt(bytes(tampered_encrypted_data), password)
            self.assertNotEqual(decrypted_message_bytes, original_message_bytes)
        except ValueError as e:
            self.assertTrue("padding" in str(e).lower() or "decryption" in str(e).lower())

    def test_aes_decrypt_tampered_iv(self):
        password = "aes_password_tamper_iv"
        original_message_bytes = b"IV secret."
        encrypted_data = aes_encrypt(original_message_bytes, password)
        tampered_encrypted_data = bytearray(encrypted_data)
        if len(tampered_encrypted_data) >= 32:
            tamper_idx = random.randint(16, 31)
            original_byte = tampered_encrypted_data[tamper_idx]
            tampered_encrypted_data[tamper_idx] = original_byte ^ 0x01
            self.assertNotEqual(tampered_encrypted_data[tamper_idx], original_byte)
        else:
            self.fail("Encrypted data too short to tamper IV.")
        try:
            decrypted_message_bytes = aes_decrypt(bytes(tampered_encrypted_data), password)
            self.assertNotEqual(decrypted_message_bytes, original_message_bytes)
        except ValueError as e:
            logging.debug(f"Tampered IV test correctly failed with ValueError: {e}") # Use pre-imported logging
            self.assertTrue("padding" in str(e).lower() or "decryption" in str(e).lower())

    # --- CRC32 Tests ---
    def test_crc32_basic(self):
        data = b"Hello World"
        expected_crc = zlib.crc32(data) & 0xFFFFFFFF # Ensure it's positive
        self.assertEqual(compute_crc32(data), expected_crc)

    def test_crc32_empty_string(self):
        data = b""
        expected_crc = zlib.crc32(data) & 0xFFFFFFFF
        self.assertEqual(compute_crc32(data), expected_crc)

    def test_crc32_consistency(self):
        data = b"Test data for consistency"
        crc1 = compute_crc32(data)
        crc2 = compute_crc32(data)
        self.assertEqual(crc1, crc2)

    def test_crc32_different_data(self):
        data1 = b"Data string 1"
        data2 = b"Data string 2"
        self.assertNotEqual(compute_crc32(data1), compute_crc32(data2))

if __name__ == '__main__':
    # If you want to see debug logs from tests, configure logging here for the test run
    # logging.basicConfig(level=logging.DEBUG)
    unittest.main()
