import numpy as np
from PIL import Image

def generate_textured_image(filename="textured_test_image.png", width=256, height=256):
  """
  Generates a PNG image with random noise (texture).

  Args:
    filename: The name of the image file to create.
    width: The width of the image in pixels.
    height: The height of theimage in pixels.
  """
  try:
    # Create a 3D numpy array (height, width, channels) with random unsigned 8-bit integers (0-255)
    random_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)

    # Create an image from the numpy array
    img = Image.fromarray(random_array, 'RGB')

    img.save(filename)
    print(f"Textured image '{filename}' created successfully.")
  except ImportError as e:
    print(f"Error: A required library is not installed: {e}. Please install it.")
    if "numpy" in str(e).lower():
        print("Try running: pip install numpy")
    if "PIL" in str(e).lower() or "Pillow" in str(e).lower():
        print("Try running: pip install Pillow")
  except Exception as e:
    print(f"An error occurred: {e}")

if __name__ == "__main__":
  generate_textured_image()
