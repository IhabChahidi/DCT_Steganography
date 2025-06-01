from PIL import Image

def generate_image(filename="test_image.png", width=256, height=256, color="blue"):
  """
  Generates a PNG image with a solid color.

  Args:
    filename: The name of the image file to create.
    width: The width of the image in pixels.
    height: The height of the image in pixels.
    color: The color of the image (e.g., "blue", "red", "#FF0000").
  """
  try:
    img = Image.new("RGB", (width, height), color)
    img.save(filename)
    print(f"Image '{filename}' created successfully.")
  except ImportError:
    print("Pillow library not found. Please install it by running: pip install Pillow")
  except Exception as e:
    print(f"An error occurred: {e}")

if __name__ == "__main__":
  generate_image()
