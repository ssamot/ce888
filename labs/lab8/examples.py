from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
import scipy.misc



nb_classes = 10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
n = 4
X_train = X_train[n:n+1,:]
y_train = y_train[n:n+1,:]
print X_train.shape
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images


# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)



for i, (X_batch, Y_batch) in enumerate(datagen.flow(X_train, Y_train, batch_size=32)):
    image =  X_batch[0].transpose(1,2,0)
    scipy.misc.imsave('./dg/image-%d.png'%(i,), image)
    if(i > 16):
        break