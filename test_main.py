import unittest
import random # For introducing errors
import os # For os.urandom in tests
import zlib # For checking CRC32 value
import shutil # For setUp and tearDown
from PIL import Image
import numpy as np

from main import (
    hamming_encode, hamming_decode,
    encode_data_with_hamming, decode_data_with_hamming,
    derive_key, aes_encrypt, aes_decrypt,
    compute_crc32, prepare_message_for_embedding,
    recover_message_from_extracted_payload,
    embed, extract
)

class TestUtilities(unittest.TestCase):

    def setUp(self):
        self.test_dir = "test_images_temp_integrations"
        os.makedirs(self.test_dir, exist_ok=True)
        self.input_image_path = os.path.join(self.test_dir, "input.png")
        self.output_image_path = os.path.join(self.test_dir, "output.png")
        self._create_dummy_image(self.input_image_path, width=128, height=128) # Larger for more K values

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def _create_dummy_image(self, filepath, width=64, height=64):
        # Ensure dimensions are multiples of 8
        width = (width // 8) * 8
        height = (height // 8) * 8
        if width == 0: width = 8
        if height == 0: height = 8

        image_array = np.zeros((height, width, 3), dtype=np.uint8)
        for y in range(height):
            for x in range(width):
                image_array[y, x] = [(x * 255) // width, (y * 255) // height, (x + y) * 255 // (width + height)]
        img = Image.fromarray(image_array, 'RGB')
        img.save(filepath, "PNG")

    # --- Hamming Coding Tests ---
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
        derived_key = derive_key(password, salt, key_len)
        self.assertEqual(len(derived_key), key_len)

        key_len_short = 16
        derived_key_short = derive_key(password, salt, key_len_short)
        self.assertEqual(len(derived_key_short), key_len_short)

    def test_derive_key_deterministic(self):
        password = "deterministic_pass"
        salt = b'fixed_salt_12345'
        key_len = 32
        key1 = derive_key(password, salt, key_len)
        key2 = derive_key(password, salt, key_len)
        self.assertEqual(key1, key2)

    def test_derive_key_salt_difference(self):
        password = "common_password"
        salt1 = os.urandom(16)
        salt2 = os.urandom(16)
        while salt1 == salt2:
            salt2 = os.urandom(16)
        key_len = 32
        key1 = derive_key(password, salt1, key_len)
        key2 = derive_key(password, salt2, key_len)
        self.assertNotEqual(key1, key2)

    def test_derive_key_password_difference(self):
        password_a = "password_A"
        password_b = "password_B"
        salt = os.urandom(16)
        key_len = 32
        key_a = derive_key(password_a, salt, key_len)
        key_b = derive_key(password_b, salt, key_len)
        self.assertNotEqual(key_a, key_b)

    # --- AES Encryption/Decryption Tests ---
    def test_aes_encrypt_decrypt_basic(self):
        password = "aes_password"
        original_message_str = "Hello, AES!"
        original_message_bytes = original_message_str.encode('utf-8')

        encrypted_data = aes_encrypt(original_message_bytes, password)
        self.assertTrue(len(encrypted_data) > 32)

        decrypted_message_bytes = aes_decrypt(encrypted_data, password)
        self.assertEqual(decrypted_message_bytes, original_message_bytes)
        self.assertEqual(decrypted_message_bytes.decode('utf-8'), original_message_str)

    def test_aes_encryption_produces_unique_outputs(self):
        password = "same_password_unique_cipher"
        message_bytes = b"Test message for uniqueness"

        encrypted_data1 = aes_encrypt(message_bytes, password)
        encrypted_data2 = aes_encrypt(message_bytes, password)

        self.assertNotEqual(encrypted_data1, encrypted_data2)

        decrypted_message1 = aes_decrypt(encrypted_data1, password)
        decrypted_message2 = aes_decrypt(encrypted_data2, password)

        self.assertEqual(decrypted_message1, message_bytes)
        self.assertEqual(decrypted_message2, message_bytes)

    def test_aes_decrypt_with_wrong_password(self):
        password = "correct_password"
        wrong_password = "wrong_password"
        message_bytes = b"Sensitive Data"

        encrypted_data = aes_encrypt(message_bytes, password)

        with self.assertRaises(ValueError):
            aes_decrypt(encrypted_data, wrong_password)

    def test_aes_encrypt_decrypt_empty_message(self):
        password = "empty_message_pass"
        original_message_bytes = b""

        encrypted_data = aes_encrypt(original_message_bytes, password)
        decrypted_message_bytes = aes_decrypt(encrypted_data, password)

        self.assertEqual(decrypted_message_bytes, original_message_bytes)

    def test_aes_encrypt_decrypt_long_message(self):
        password = "long_message_pass"
        original_message_str = "This is a much longer message designed to span multiple AES blocks. " * 20
        original_message_bytes = original_message_str.encode('utf-8')

        encrypted_data = aes_encrypt(original_message_bytes, password)
        decrypted_message_bytes = aes_decrypt(encrypted_data, password)

        self.assertEqual(decrypted_message_bytes, original_message_bytes)
        self.assertEqual(decrypted_message_bytes.decode('utf-8'), original_message_str)

    # --- CRC32 Test ---
    def test_compute_crc32_basic(self):
        data_bytes = b"hello world"
        expected_crc = zlib.crc32(data_bytes) & 0xFFFFFFFF
        self.assertEqual(compute_crc32(data_bytes), expected_crc)
        self.assertEqual(compute_crc32(data_bytes), 222957957)

    # --- Message Preparation and Recovery Tests ---
    def _common_prepare_recover_test(self, message_str, password):
        final_bits_for_embedding, L_payload_bits = prepare_message_for_embedding(message_str, password)
        encoded_len_part = final_bits_for_embedding[:28]
        encoded_payload_part = final_bits_for_embedding[28:]
        recovered_decoded_len_bits = decode_data_with_hamming(encoded_len_part)
        self.assertEqual(len(recovered_decoded_len_bits), 16)
        extracted_L_payload_from_header = int("".join(map(str, recovered_decoded_len_bits)), 2)
        self.assertEqual(extracted_L_payload_from_header, L_payload_bits)
        recovered_decoded_payload_bits = decode_data_with_hamming(encoded_payload_part)
        self.assertTrue(len(recovered_decoded_payload_bits) >= L_payload_bits)
        recovered_message = recover_message_from_extracted_payload(
            recovered_decoded_len_bits,
            recovered_decoded_payload_bits,
            password
        )
        self.assertEqual(recovered_message, message_str)

    def test_prepare_and_recover_message_basic(self):
        self._common_prepare_recover_test("Hello Steganography!", "testRecoveryPass")

    def test_prepare_and_recover_message_empty(self):
        self._common_prepare_recover_test("", "emptyMsgPass")

    def test_prepare_and_recover_message_long(self):
        long_msg = "This is a longer test message to ensure everything works smoothly with more data. " * 10
        self._common_prepare_recover_test(long_msg, "longMsgRecoveryPass")

    def test_recover_message_crc_error_simulation(self):
        original_message_str = "Test CRC Error"
        password = "crcErrorPassword"
        message_bytes = original_message_str.encode('utf-8')
        actual_crc = compute_crc32(message_bytes)
        bad_crc = (actual_crc + 1) % (2**32)
        message_with_bad_crc = message_bytes + bad_crc.to_bytes(4, 'big')
        encrypted_payload_with_bad_crc_bytes = aes_encrypt(message_with_bad_crc, password)
        payload_bits_with_bad_crc = [int(b) for byte in encrypted_payload_with_bad_crc_bytes for b in format(byte, '08b')]
        L_payload_bad = len(payload_bits_with_bad_crc)
        len_bits_for_header_bad_decoded = [int(b) for b in bin(L_payload_bad)[2:].zfill(16)]
        with self.assertRaisesRegex(ValueError, "CRC mismatch"):
            recover_message_from_extracted_payload(
                len_bits_for_header_bad_decoded,
                payload_bits_with_bad_crc,
                password
            )

    # --- Integration Tests (embed and extract) ---
    def test_embed_extract_basic(self):
        message = "Integration Test Message!"
        key = 12345
        # Using K=50 as K=10 might be too low for robust recovery depending on image and alpha
        # Alpha=0.5 is a moderate strength
        embed(self.input_image_path, message, self.output_image_path, key, alpha=0.5, K=50)
        extracted_message = extract(self.output_image_path, key, K=50)
        self.assertEqual(extracted_message, message)

    def test_embed_extract_different_keys(self):
        message = "Don't tell anyone!"
        key1 = 11111
        key2 = 22222
        embed(self.input_image_path, message, self.output_image_path, key1, alpha=0.5, K=50)
        # Expecting ValueError from recover_message_from_extracted_payload (CRC or AES padding)
        with self.assertRaises(ValueError):
            extract(self.output_image_path, key2, K=50)

    def test_embed_extract_message_too_long(self):
        message = "This message is way too long for the system's current configuration. " * 5 # ~175 chars
        key = 67890
        with self.assertRaisesRegex(ValueError, "Message too long"):
            embed(self.input_image_path, message, self.output_image_path, key, alpha=0.5, K=50)

    def test_embed_non_existent_input_image(self):
        with self.assertRaises(FileNotFoundError):
            embed("non_existent_image.png", "msg", self.output_image_path, 123, alpha=0.1, K=10)

    def test_extract_non_existent_input_image(self):
        with self.assertRaises(FileNotFoundError):
            extract("non_existent_image.png", 123, K=10)

    def test_embed_invalid_image_format_file_ext(self):
        # Create a dummy text file
        dummy_txt_path = os.path.join(self.test_dir, "invalid.txt")
        with open(dummy_txt_path, "w") as f:
            f.write("This is not an image.")
        with self.assertRaisesRegex(ValueError, "Image must be PNG or JPEG"):
            embed(dummy_txt_path, "msg", self.output_image_path, 123, alpha=0.1, K=10)

    def test_extract_invalid_image_format_file_ext(self):
        dummy_txt_path = os.path.join(self.test_dir, "invalid_for_extract.txt")
        with open(dummy_txt_path, "w") as f:
            f.write("This is not an image.")
        with self.assertRaisesRegex(ValueError, "Image must be PNG or JPEG"):
            extract(dummy_txt_path, 123, K=10)

    # Consider adding a test for image content that isn't valid DCT input if possible,
    # but current checks mostly look at format via extension or if cv2.imread returns None.
    # For example, a PNG that is corrupted. This is harder to set up reliably.

if __name__ == '__main__':
    unittest.main()
