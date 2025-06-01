from PIL import Image, ImageDraw

# Create a new image with a white background
img = Image.new('RGB', (64, 64), color = 'white')
draw = ImageDraw.Draw(img)

# Draw a red square
draw.rectangle([(10, 10), (30, 30)], fill='red')
# Draw a green square
draw.rectangle([(34, 10), (54, 30)], fill='green')
# Draw a blue square
draw.rectangle([(10, 34), (30, 54)], fill='blue')
# Draw a black square
draw.rectangle([(34, 34), (54, 54)], fill='black')

img.save('test_image.png')
print("Created test_image.png")
