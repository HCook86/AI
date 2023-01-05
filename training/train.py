# SOURCE 
#   Data Source: http://yann.lecun.com/exdb/mnist/
#   Reading software source: https://stackoverflow.com/questions/42812230/why-plt-imshow-doesnt-display-the-image
#   idx2numpy --> https://github.com/ivanyu/idx2numpy
#   matplotlib --> https://matplotlib.org/
#   numpy -->  https://numpy.org/doc/stable/

import numpy as np
import idx2numpy
import matplotlib.pyplot as plt

imagefile = 'test/t10k-images-idx3-ubyte'
imagearray = idx2numpy.convert_from_file(imagefile)

labelsfile = 'test/t10k-labels-idx1-ubyte'
labelsarray = idx2numpy.convert_from_file(labelsfile)


print(labelsarray)

#In decimal. Range 0-1
processed_imagearray = (1/255)*imagearray

# List of 784*1 numpy arrays (vector) containing number data
images = list()
for i in processed_imagearray:
    images.append(i.flatten())

# Final training data
data = list()
# Transfrom to the following data structure: (np.array(784*1), int)
for i in range(0, len(images)):
    data.append((images[i],labelsarray[i]))


plt.imshow(processed_imagearray[9999], cmap=plt.cm.binary)
plt.show()
