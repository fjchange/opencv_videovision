# videotransform_opencv
This repo is opencv accelerated video tensorform lib for pytorch. Part of this repo is modified from [opencv_transforms](https://github.com/jbohnslav/opencv_transforms) and [torch_videovision](https://github.com/hassony2/torch_videovision). 

As descirbed in [opencv_transforms](https://github.com/jbohnslav/opencv_transforms), opencv is faster than PIL in most cases, while torchvision offer a PIL based transforms to avoid BGR or RGB confusion(? refer to the [issue](https://github.com/pytorch/vision/pull/34) in torchvision repo.) 

Thanks jbohnslav's work in rewrite transform in opencv based, this repo is to expand it to video domain.
