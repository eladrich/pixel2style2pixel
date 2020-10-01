<p align="center">
<img src="https://github.com/eladrich/pixel2style2pixel/blob/master/docs/teaser.jpg?raw=true" width="800px"/>
</p>

> We present a generic image-to-image translation framework, Pixel2Style2Pixel (pSp). Our pSp framework is based on a novel encoder network that directly generates a series of style vectors which are fed into a pretrained StyleGAN generator, forming the extended W+ latent space. We first show that our encoder can directly embed real images into W+, with no additional optimization. We further introduce a dedicated identity loss which is shown to achieve improved performance in the reconstruction of an input image. We demonstrate pSp to be a simple architecture that, by leveraging a well-trained, fixed generator network, can be easily applied on a wide-range of image-to-image translation tasks. Solving these tasks through the style representation results in a global approach that does not rely on a local pixel-to-pixel correspondence and further supports multi-modal synthesis via the resampling of styles. Notably, we demonstrate that pSp can be trained to align a face image to a frontal pose without any labeled data, generate multi-modal results for ambiguous tasks such as conditional face generation from segmentation maps, and construct high-resolution images from corresponding low-resolution images.



## Applications
### StyleGAN Encoding
Here, we use pSp to find the latent code of real images in the latent domain of a pretrained StyleGAN generator. 
<p align="center">
<img src="https://media.giphy.com/media/cvTXRcBfkXpmRjccCE/giphy-downsized-large.gif" width="800px"/>

</p>


### Face Frontalization
In this application we want to generate a front-facing face from a given input image. 
<p align="center">
<img src="https://media.giphy.com/media/cOT6P3dYVN6ILOd1uP/source.gif" width="800px"/>

</p>

### Conditional Image Synthesis
Here we wish to generate photo-realistic face images from ambiguous sketch images or segmentation maps. Using style-mixing, we inherently support mutli-modal synthesis for a single input.
<p align="center">
<img src="https://media.giphy.com/media/p2nWmurqzZoUTbEwfL/giphy.gif" width="800px"/>

</p>

### Super Resolution
Given a low-resolution input image, we generate a corresponding high-resolution image. As this too is an ambiguous task, we can use style-mixing to produce several plausible results.
<p align="center">
<img src="https://media.giphy.com/media/ijEUoHX5zWq6gdDHpg/giphy-downsized-large.gif" width="800px"/>

</p>
