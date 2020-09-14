import numbers
import random
import numpy as np
import math
import collections
import torch
from . import functional as F
import cv2
import warnings
import types

__all__ =['Compose','ToTensor','ClipToTensor','Lambda','Normalize','Resize','RandomCrop','CenterCrop','RandomHorizontalFlip'
    ,'RandomVerticalFlip','RandomResizedCrop','TenCrop','ColorJitter','RandomRotation','RandomGrayScale']

_cv2_pad_to_str = {'constant':cv2.BORDER_CONSTANT,
                   'edge':cv2.BORDER_REPLICATE,
                   'reflect':cv2.BORDER_REFLECT_101,
                   'symmetric':cv2.BORDER_REFLECT
                  }
_cv2_interpolation_to_str= {'nearest':cv2.INTER_NEAREST,
                         'bilinear':cv2.INTER_LINEAR,
                         'area':cv2.INTER_AREA,
                         'bicubic':cv2.INTER_CUBIC,
                         'lanczos':cv2.INTER_LANCZOS4}
_cv2_interpolation_from_str= {v:k for k,v in _cv2_interpolation_to_str.items()}

class Compose(object):
    """Composes several transforms

    Args:
    transforms (list of ``Transform`` objects): list of transforms
    to compose
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, clip):
        for t in self.transforms:
            clip = t(clip)
        return clip

class ToTensor(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        return F.to_tensor(pic)

    def __repr__(self):
        return self.__class__.__name__ + '()'

class ClipToTensor(object):
    """Convert a list of m (H x W x C) numpy.ndarrays in the range [0, 255]
        to a torch.FloatTensor of shape (C x m x H x W) in the range [0, 1.0]
    """

    def __init__(self, channel_nb=3, div_255=True):
        self.channel_nb = channel_nb
        self.div_255 = div_255

    def __call__(self, clip):
        """
        Args: clip (list of numpy.ndarray)
        to be converted to tensor.
        """
        # Retrieve shape
        if isinstance(clip[0], np.ndarray):
            h, w, ch = clip[0].shape
            assert ch == self.channel_nb, 'Got {0} instead of 3 channels'.format(
                ch)
        else:
            raise TypeError('Expected numpy.ndarray, not recommend PIL.Image as it is slow\
            but got list of {0}'.format(type(clip[0])))

        clip=np.stack(clip,axis=0).astype(float)
        if self.div_255:
            clip = clip / 255.
        clip=torch.from_numpy(clip).float()
        clip=clip.permute([3,0,1,2])
        return clip

class Lambda(object):
    """Apply a user-defined lambda as a transform.
    Args:
        lambd (function): Lambda/function to be used for transform.
    """

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'

class Normalize(object):
    """Normalize a tensor image with mean and standard deviation
    Given mean: m and std: s
    will  normalize each channel as channel = (channel - mean) / std
    Args:
        mean (list of float): mean value for each channel
        std (list of float): std value for each channel
        format (str): meaning of each dim
    """

    def __init__(self, mean, std,format='CTHW'):
        self.mean = mean
        self.std = std
        self.format = format

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor of stacked images or image
            of size (C, T, H, W) to be normalized
        Returns:
            Tensor: Normalized stack of tensor of image
        """
        if self.format=='CTHW':
            tensor=tensor.permute([1,2,3,0])
            for i in range(tensor.shape[0]):
                tensor[i]=F.normalize(tensor[i], self.mean, self.std)
            tensor=tensor.permute([3,0,1,2])
        elif self.format=='THWC':
            for i in range(tensor.shape[0]):
                tensor[i]=F.normalize(tensor[i], self.mean, self.std)
        else:
            raise ValueError('Expected format in between [CTWH, THWC]')
        return tensor

class Resize(object):
    """Resize the list of input numpy ndarray to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``cv2.INTER_CUBIC``, bicubic interpolation
    """

    def __init__(self, size, interpolation='bilinear'):
        # assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        if isinstance(size, int):
            self.size = (size,size)
        elif isinstance(size, collections.Iterable) and len(size) == 2:
            if type(size) == list:
                size = tuple(size)
            self.size = size
        else:
            raise ValueError('Unknown inputs for size: {}'.format(size))
        self.interpolation = _cv2_interpolation_to_str[interpolation]

    def __call__(self, clip):
        """
        Args:
            clip ( list of numpy ndarray): Image to be scaled.
        Returns:
            numpy ndarray: Rescaled image.
        """
        return [F.resize(img, self.size, self.interpolation) for img in clip]

    def __repr__(self):
        interpolate_str = _cv2_interpolation_from_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)

class RandomCrop(object):
    """Crops a random spatial crop in a list of images input [Time, Height, Width, Channel]
       """

    def __init__(self, size):
        """
        Args:
            size (tuple): in format (height, width)
        """
        self.size = size

    def __call__(self, clip):
        h, w = self.size
        img_h, img_w,_ = clip[0].shape

        if w > img_w or h > img_h:
            error_msg = (
                'Initial tensor spatial size should be larger then '
                'cropped size but got cropped sizes : ({w}, {h}) while '
                'initial tensor is ({t_w}, {t_h})'.format(
                    t_w=img_w, t_h=img_h, w=w, h=h))
            raise ValueError(error_msg)
        x1 = random.randint(0, img_w - w)
        y1 = random.randint(0, img_h - h)
        cropped = [F.crop(img,y1,x1,h,w)for img in clip]
        return cropped

class CenterCrop(object):
    """Crops the given numpy ndarray at the center.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, clip):
        """
        Args:
            img (numpy ndarray): Image to be cropped.
        Returns:
            numpy ndarray: Cropped image.
        """
        return [F.center_crop(img, self.size)for img in clip]

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)

class RandomHorizontalFlip(object):
    """Horizontally flip the list of given images randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, clip):
        """
        Args:
        img (Opencv Image or numpy.ndarray): List of images to be cropped
        in format (h, w, c) in numpy.ndarray

        Returns:
        numpy.ndarray: Randomly flipped clip
        """
        if random.random() < self.p:
            if isinstance(clip[0], np.ndarray):
                return [F.hflip(img) for img in clip]
            else:
                raise TypeError('Expected numpy.ndarray, not recommend PIL.Image as it is slow' +
                                ' but got list of {0}'.format(type(clip[0])))
        return clip

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class RandomVerticalFlip(object):
    """Horizontally flip the list of given images randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, clip):
        """
        Args:
        clip (Opencv Images or numpy.ndarray): List of images to be cropped
        in format (h, w, c) in numpy.ndarray

        Returns:
        numpy.ndarray: Randomly flipped clip
        """
        if random.random() < self.p:
            if isinstance(clip[0], np.ndarray):
                return [F.vflip(img) for img in clip]
            else:
                raise TypeError('Expected numpy.ndarray, not recommend PIL.Image as it is slow' +
                                ' but got list of {0}'.format(type(clip[0])))
        return clip

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class RandomResizedCrop(object):
    """Crop the given list of PIL Images to random size and aspect ratio.
    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.
    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation='bilinear'):
        if isinstance(size, (tuple, list)):
            self.size = size
        else:
            self.size = (size, size)
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")

        self.interpolation = _cv2_interpolation_to_str[interpolation]
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(clip, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.
        Args:
            img (list of PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        if isinstance(clip[0], np.ndarray):
            height, width, im_c = clip[0].shape
        else:
            raise TypeError('Expected numpy.ndarray, not recommend PIL.Image as it is slow')
        area = height * width

        for _ in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if (in_ratio < min(ratio)):
            w = width
            h = int(round(w / min(ratio)))
        elif (in_ratio > max(ratio)):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def __call__(self, clip):
        """
        Args:
            clip:list of np.ndarray: Image to be cropped and resized.
        Returns:
            list of np.ndarray: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(clip, self.scale, self.ratio)
        clip = [F.resized_crop(img, i, j, h, w,self.size,self.interpolation) for img in clip]

        return clip

    def __repr__(self):
        interpolate_str = _cv2_interpolation_to_str[self.interpolation]
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
        format_string += ', interpolation={0})'.format(interpolate_str)
        return format_string

class TenCrop(object):
    """Crop the list of numpy arrays to into four corners and the central crop plus the flipped version of
    these (horizontal flipping is used by default)
    ..Note:
    The output would be with size of [10,T,H,W,C]
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        vertical_flip(bool): Use vertical flipping instead of horizontal
    Example:
    #     >>> transform = Compose([
    #     >>>    TenCrop(size), # this is a list of PIL Images
    #     >>>    Lambda(lambda crops: torch.stack([ClipToTensor()(crop) for crop in crops])) # returns a 4D tensor
    #     >>> ])
    #     >>> #In your test loop you can do the following:
    #     >>> input, target = batch # input is a 6d tensor, target is 2d
    #     >>> bs, ncrops, c, t, h, w = input.size()
    #     >>> result = model(input.view(-1, c, t, h, w)) # fuse batch size and ncrops
    #     >>> result_avg = result.view(bs, ncrops, -1).mean(1) # avg over crops
    """
    def __init__(self,size,vertical_flip=False):
        self.size = size
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            assert len(size) == 2, "Please provide only two dimensions (h, w) for size."
            self.size = size
        self.vertical_flip = vertical_flip
    def __call__(self,clip):
        return [F.ten_crop(img, self.size, self.vertical_flip) for img in clip]

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, vertical_flip={1})'.format(self.size, self.vertical_flip)

class RandomGrayScale(object):
    """Randomly convert image to grayscale with a probability of p (default 0.1).
    The image can be a PIL Image or a Tensor, in which case it is expected
    to have [..., 3, H, W] shape, where ... means an arbitrary number of leading
    dimensions
    Args:
        p (float): probability that image should be converted to grayscale.
    Returns:
        PIL Image or Tensor: Grayscale version of the input image with probability p and unchanged
        with probability (1-p).
        - If input image is 1 channel: grayscale version is 1 channel
        - If input image is 3 channel: grayscale version is 3 channel with r == g == b
    """
    def __init__(self, p=0.1):
        super().__init__()
        self.p = p
    def __call__(self,clip):
        """
        Args:
            list of imgs (PIL Image or Tensor): Image to be converted to grayscale.
        Returns:
            PIL Image or Tensor: Randomly grayscaled image.
        """
        num_output_channels = 1 if len(clip[0].shape)==2 else 3
        if torch.rand(1)<self.p:
            for i in range(len(clip)):
                clip[i]=F.to_grayscale(clip[i],num_output_channels)
        return clip

class ColorJitter(object):
    """Randomly change the brightness, contrast and saturation of an image.
    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)
        if self.saturation is not None:
            warnings.warn('Saturation jitter enabled. Will slow down loading immensely.')
        if self.hue is not None:
            warnings.warn('Hue jitter enabled. Will slow down loading immensely.')

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with length 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        """Get a randomized transform to be applied on image.
        Arguments are same as that of __init__.
        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []

        if brightness is not None:
            brightness_factor = random.uniform(brightness[0], brightness[1])
            transforms.append(Lambda(lambda img: F.adjust_brightness(img, brightness_factor)))

        if contrast is not None:
            contrast_factor = random.uniform(contrast[0], contrast[1])
            transforms.append(Lambda(lambda img: F.adjust_contrast(img, contrast_factor)))

        if saturation is not None:
            saturation_factor = random.uniform(saturation[0], saturation[1])
            transforms.append(Lambda(lambda img: F.adjust_saturation(img, saturation_factor)))

        if hue is not None:
            hue_factor = random.uniform(hue[0], hue[1])
            transforms.append(Lambda(lambda img: F.adjust_hue(img, hue_factor)))

        random.shuffle(transforms)
        transform = Compose(transforms)

        return transform

    def __call__(self, clip):
        """
        Args:
            clip (list of numpy ndarray): Input images.
        Returns:
            numpy ndarray: Color jittered image.
        """
        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)

        return [transform(img) for img in clip]

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        return format_string

class RandomRotation(object):
    """Rotate the image by angle.
    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees).
        resample ({cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4}, optional):
            An optional resampling filter. See `filters`_ for more information.
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
    """

    def __init__(self, degrees, resample=False, expand=False, center=None):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.resample = resample
        self.expand = expand
        self.center = center

    @staticmethod
    def get_params(degrees):
        """Get parameters for ``rotate`` for a random rotation.
        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        angle = random.uniform(degrees[0], degrees[1])

        return angle

    def __call__(self, clip):
        """
            clip ( list of numpy ndarray): Images to be rotated.
        Returns:
            numpy ndarray: Rotated image.
        """

        angle = self.get_params(self.degrees)

        return [F.rotate(img, angle, self.resample, self.expand, self.center) for img in clip]

    def __repr__(self):
        format_string = self.__class__.__name__ + '(degrees={0}'.format(self.degrees)
        format_string += ', resample={0}'.format(self.resample)
        format_string += ', expand={0}'.format(self.expand)
        if self.center is not None:
            format_string += ', center={0}'.format(self.center)
        format_string += ')'
        return format_string
