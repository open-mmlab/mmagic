import numpy as np
import torch
from mmcv.cnn import kaiming_init, normal_init, xavier_init
from torch.nn import init


def generation_init_weights(module, init_type='normal', init_gain=0.02):
    """Default initialization of network weights for image generation.

    By default, we use 'normal' init, but xavier and kaiming might work
    better for some applications.

    Args:
        module (nn.Module): Module to be initialized.
        init_type (str): The name of an initialization method:
            normal | xavier | kaiming | orthogonal
        init_gain (float): Scaling factor for normal, xavier and
            orthogonal.
    """

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1
                                     or classname.find('Linear') != -1):
            if init_type == 'normal':
                normal_init(m, 0.0, init_gain)
            elif init_type == 'xavier':
                xavier_init(m, gain=init_gain, distribution='normal')
            elif init_type == 'kaiming':
                kaiming_init(
                    m,
                    a=0,
                    mode='fan_in',
                    nonlinearity='leaky_relu',
                    distribution='normal')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight, gain=init_gain)
                init.constant_(m.bias.data, 0.0)
            else:
                raise NotImplementedError(
                    f"Initialization method '{init_type}' is not implemented")
        elif classname.find('BatchNorm2d') != -1:
            # BatchNorm Layer's weight is not a matrix;
            # only normal distribution applies.
            normal_init(m, 1.0, init_gain)

    module.apply(init_func)


class GANImageBuffer(object):
    """This class implements an image buffer that stores previously
    generated images.

    This buffer allows us to update the discriminator using a history of
    generated images rather than the ones produced by the latest generator
    to reduce model oscillation.

    Args:
        buffer_size (int): The size of image buffer. If buffer_size = 0,
            no buffer will be created.
        buffer_ratio (float): The chance / possibility  to use the images
            previously stored in the buffer.
    """

    def __init__(self, buffer_size, buffer_ratio=0.5):
        self.buffer_size = buffer_size
        # create an empty buffer
        if self.buffer_size > 0:
            self.img_num = 0
            self.image_buffer = []
        self.buffer_ratio = buffer_ratio

    def query(self, images):
        if self.buffer_size == 0:  # if the buffer size is 0, do nothing
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            # if the buffer is not full, keep inserting current images
            if self.img_num < self.buffer_size:
                self.img_num = self.img_num + 1
                self.image_buffer.append(image)
                return_images.append(image)
            else:
                use_buffer = np.random.random() < self.buffer_ratio
                # by self.buffer_ratio, the buffer will return a previously
                # stored image, and insert the current image into the buffer
                if use_buffer:
                    random_id = np.random.randint(0, self.buffer_size)
                    image_tmp = self.image_buffer[random_id].clone()
                    self.image_buffer[random_id] = image
                    return_images.append(image_tmp)
                # by (1 - self.buffer_ratio), the buffer will return the
                # current image
                else:
                    return_images.append(image)
        # collect all the images and return
        return_images = torch.cat(return_images, 0)
        return return_images
