import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from itertools import product


# IMAGE PROCESSING USING CNN

# setting parameters in matplotlib Autolayout ensures figures adjust automatically for better readability. Sets colormap (magma) for better contrast in images.

plt.rc('figure',autolayout=True)
plt.rc('image',cmap='magma')

# defining convolutional kernel (edge detection)
# This 3x3 kernel (Laplacian filter) is used for edge detection. It highlights regions of rapid intensity change by assigning high weights to the center pixel and negative weights to the surrounding pixels.

kernel = tf.constant([[-1,-1,-1],
                      [-1,8,-1],
                      [-1,-1,-1]])

# loading and processing the image by reading it, decoding jpeg to tensorflow tensor (grayscale) and resizing it to image of 300x300

image = tf.io.read_file('images/superman_hope.jpg')
image = tf.io.decode_jpeg(image,channels=1) # channels=1 converts to grayscale
image = tf.image.resize(image,size=[300,300])

# displaying original image
img = tf.squeeze(image).numpy()

plt.subplot(1,4,1)
plt.imshow(img,cmap='gray')
plt.axis('off')
plt.title('Grayscale Image')
# plt.show()

# reformatting image and kernel for tensorflow processing

image = tf.image.convert_image_dtype(image,dtype=tf.float32) #converting image to float32 for numerical stability
image = tf.expand_dims(image, axis=0) # Expands dimensions to match the expected input format for TensorFlow's convolution operation: (Batch Size, Height, Width, Channels) → (1, 300, 300, 1).
kernel = tf.reshape(kernel,[*kernel.shape,1,1]) # Reshapes the kernel from (3,3) to (3, 3, 1, 1), so it works with 1-channel (grayscale) images. 1 (third dimension) → Matches the input image channels (grayscale = 1). 1 (fourth dimension) → Number of filters (applying one filter in this case).
kernel = tf.cast(kernel,dtype=tf.float32) # converting kernel to float32


# Applying 2D convolution (edge detection) to the image
# Effect: Highlights edges by computing local changes in pixel intensity.

conv_fn = tf.nn.conv2d
image_filter = conv_fn(
    input=image,
    filters=kernel,
    strides=1, #Skipping no pixels by moving 1 pixel at a time.
    padding='SAME' # keeping output size same as input
)

# Displaying covolved image

plt.subplot(1,4,2)
plt.imshow(tf.squeeze(image_filter))
plt.axis('off')
plt.title('Covolution')
# plt.show()

# Applying Activation function (ReLU
#Applies Rectified Linear Unit (ReLU) activation, which: Keeps positive values as they are. Converts negative values to 0 (removes unnecessary details). Effect: Enhances edges by removing negative values.

relu_fn = tf.nn.relu
image_detect = relu_fn(image_filter)

#Displaying activated image

plt.subplot(1,4,3)
plt.imshow(tf.squeeze(image_detect))
plt.axis('off')
plt.title('Activation')
# plt.show()

# Applying Pooling (Downsampling)
# This reduces image size by half by retaining important features, hence enhancing the details of previous activated imaged
pool = tf.nn.pool
image_condensed = pool(input=image_detect,
                      window_shape=(2,2),
                      pooling_type='MAX',
                      strides=(2,2), # takes maximum value from each (2x2) region
                      padding='SAME')

# displaiyng pooled image

plt.subplot(1,4,4)
plt.imshow(tf.squeeze(image_condensed))
plt.axis('off')
plt.title('Pooling')
plt.show()
