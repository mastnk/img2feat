# img2feat

It is an image feature extractor based on a convolutional neural network.

[Github](https://github.com/mastnk/img2feat/) [PyPI](https://pypi.org/project/img2feat/)

## Installation

```% pip install img2feat
```

Required libraries: numpy, torch, torchvision, opencv-python

## *class* CNN

The CNN converts a list of numpy images to features, where numpy image is assumed opencv format, or [Height, Width, BGR].
The shape of the output features is [ length of the list of the input images, *dim_feature* of the CNN].

Available networks: 
alexnet,
vgg11, vgg13, vgg16, vgg19, 
resnet18, resnet34, resnet101, resnet152,
densenet121, densenet161, densenet169, densenet201, 
googlenet, mobilenet, 
vit_b_16

```python
from img2feat import BuildNet
net = BuildNet.build('vgg11')
x = net( [img] )
```

### Methods
- **available_networks**() -> list of string

		Return the list of names of available networks.

- **__init__**( network='vgg11', gpu=False, img_size=(224,224) )

	Constructor
 
	*network* should be one of *available_networks*()

	*gpu* is set True, if the GPU is available.

	*img_size* is the image size which is input to the network, (width, height)	

- **__call__**( imgs ) -> feature (numpy float32)

	It converts the list of images to the features.

	*imgs* is thee list of images. The image is should be the opencv format, or [Height, Width, BGR].

	*feature* is the converted features where [ len(imgs), *dim_feature*].

### Variables
- **dim_feature** (int)

	It is the dimension of the output feature.

## *class* PixelFeature

The PixelFeature converts images to per-pixel features.
The feature is the numpy array of [Height, Width, Dim Feature].

Available networks: 
vgg11, vgg13, vgg16, vgg19, 


```python
from img2feat import CNN
net = CNN('vgg11')
x = net( [img] )
```

### Methods
- **available_networks**() -> list of string

		Return the list of names of available networks.

- **__init__**( network='vgg11', gpu=False )

	Constructor
 
	*network* should be one of *available_networks*()

	*gpu* is set True, if the GPU is available.

- **__call__**( imgs ) -> list of feature (numpy float32)

	It converts the list of images to the features.

	*imgs* is thee list of images. The image is should be the opencv format, or [Height, Width, BGR].

	*feature* is the converted features where [ height, width, *dim_featute*]. The height and width are same as the input image.

### Variables
- **dim_feature** (int)

	It is the dimension of the output feature.

## *class* Mirror

The Mirror provide a data augmentation of mirroring.

### Methods

- **__call__**( imgs ) -> augmented images

	It return the augmented images. 
	The output is the list of images. The odd is the original images and the even is the mirrored images.

### Variables

- **nb_aug** int

	It return 2.

## *class* TenCrop

The TenCrop provide a typical 10-crop data augmentation.
First, images are resized so that the shorter side is a setting scale.
Then, center, top-left, top-right, bottom-left, and bottom-right are cropped.

### Methods

- **__init__**( scales=[224, 256, 384, 480, 640], mirror=True, img_size=(224,224) )

	Constructor.

	*scales* is a list of scales. Images are resized so that the shorter side is scale.

	If *mirror* is True, the mirroring augmentation is also applied.

	*img_size* is cropping size.

- **__call__**( imgs ) -> augmented images

	It returns the augmented images.

### Variables

- **img_size**

	It is the cropping size. [Width, Height]

- **nb_aug**

	It is the number of augmentation fro a single image.
	It is the multiplication of len(scales) * 5 * 2, if mirror is True

## *package* antbee

It is utility package for the dataset of ants and bees in 
[Transfer Learning for Computer Vision Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html).

### Methods

- **load**( squared=True, root=None ) -> ( Itrain, Ytrain ), ( Itest, Ytest )

	*root* is the root directory of the data. If it is None, the root directory is set as the package directory.

	If *squared* is True, only squared images are loaded.
	If *squared* is False, all images are loaded.

	*Itrain, Itest* are lists of images.

	*Ytrain, Ytest* are numpy array of the label. 0: ant, 1: bee.

- **load_squared_npy**( name, root=None ) -> ( Xtrain, Ytrain ), ( Xtest, Ytest )

	*root* is the root directory of the data. If it is None, the root directory is set as the package directory.

	*name* is the name of CNN network.
	
	*Xtrain, Xtest* are numpy array of extracted features.

	*Ytrain, Ytest* are numpy array of the label. 0: ant, 1: bee.

### Variables

- **str**

	str[0]: 'ant', str[1]: 'bee'


## Sample Codes

[sample1.py](https://github.com/mastnk/img2feat/blob/main/sample1.py): Linear regression.

[sample2.py](https://github.com/mastnk/img2feat/blob/main/sample2.py): Data augmentation.

## Network References

[AlexNet: One weird trick for parallelizing convolutional neural networks](https://arxiv.org/abs/1404.5997)

[VGG: Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)

[ResNet: Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

[DenseNet: Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)

[MobileNet: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)

[GoogLeNet: Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842)

