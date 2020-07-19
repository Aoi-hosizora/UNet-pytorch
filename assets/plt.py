import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

image_1 = Image.open('./test.jpg')
image_2 = Image.open('./test_seg.jpg')

image_1 = np.array(image_1)
image_2 = np.array(image_2)

plt.figure(figsize=(12, 5))

plt.subplot(121)
plt.imshow(image_1)
plt.axis('off')
plt.title('Source')

plt.subplot(122)
plt.imshow(image_2)
plt.axis('off')
plt.title('Target')

plt.savefig('./assets/demo.pdf')
plt.show()
