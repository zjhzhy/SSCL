# -*- coding=utf-8 -*-

from PIL import Image
import numpy as np
import torch
import os
from tools.misc import get_example_params, save_class_activation_images
from torch.autograd import Variable


class CamExtractor():
    """
        Extracts cam features from the models
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad

    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None
        for module_pos, module in self.model.features._modules.items():
            print(module_pos)
            x = module(x)  # Forward
            if int(module_pos) == self.target_layer:
                x.register_hook(self.save_gradient)
                conv_output = x  # Save the convolution output on that layer
        return conv_output, x

    def forward_pass(self, x):
        """
            Does a full forward pass on the models
        """
        # Forward pass on the convolutions
        conv_output, x = self.forward_pass_on_convolutions(x)
        # x = x.view(x.size(0), -1)  # Flatten
        # # Forward pass on the classifier
        # x = self.model.classifier(x)
        return conv_output, x


class GradCam():
    """
        Produces class activation map
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        # Define extractor
        self.extractor = CamExtractor(self.model, target_layer)

    def generate_cam(self, input_image, target_class=None):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the models (1, 1000)
        conv_output, model_output = self.extractor.forward_pass(input_image)
        if target_class is None:
            target_class = np.argmax(model_output.data.numpy())
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        # Zero grads
        self.model.features.zero_grad()
        self.model.classifier.zero_grad()
        # Backward pass with specified target
        model_output.backward(gradient=one_hot_output, retain_graph=True)
        # Get hooked gradients
        guided_gradients = self.extractor.gradients.data.numpy()[0]
        # Get convolution outputs
        target = conv_output.data.numpy()[0]
        # Get weights from gradients
        weights = np.mean(guided_gradients, axis=(1, 2))  # Take averages for each gradient
        # Create empty numpy array for cam
        cam = np.ones(target.shape[1:], dtype=np.float32)
        # Multiply each weight with its conv output and then, sum
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2],
                                                    input_image.shape[3]), Image.ANTIALIAS))
        # ^ I am extremely unhappy with this line. Originally resizing was done in cv2 which
        # supports resizing numpy matrices, however, when I moved the repository to PIL, this
        # option is out of the window. So, in order to use resizing with ANTIALIAS feature of PIL,
        # I briefly convert matrix to PIL image and then back.
        # If there is a more beautiful way, send a PR.
        return cam


def load_model(model, path=None):
    device = torch.device("cpu")
    if path:
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint['models'])
    return model.to(device)


def preprocess_image(pil_im, resize_im=True):
    """
        Processes image for CNNs
    Args:
        PIL_img (PIL_img): Image to process
        resize_im (bool): Resize to 224 or not
    returns:
        im_as_var (torch variable): Variable that contains processed float tensor
    """
    # mean and std list for channels (Imagenet)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # Resize image
    if resize_im:
        pil_im.thumbnail((512, 512))
    im_as_arr = np.float32(pil_im)
    im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
    # Normalize the channels
    for channel, _ in enumerate(im_as_arr):
        im_as_arr[channel] /= 255
        im_as_arr[channel] -= mean[channel]
        im_as_arr[channel] /= std[channel]
    # Convert to float tensor
    im_as_ten = torch.from_numpy(im_as_arr).float()
    # Add one more channel to the beginning. Tensor shape = 1,3,224,224
    im_as_ten.unsqueeze_(0)
    # Convert to Pytorch variable
    im_as_var = Variable(im_as_ten, requires_grad=True)
    return im_as_var


from torchvision import datasets, models, transforms

valid_transforms = transforms.Compose([
    transforms.RandomResizedCrop(size=224, scale=(0.7, 0.7)),
    transforms.ToTensor(),
])


def get_cam(model, target_layer, img_path, target_class):
    finename = os.path.basename(img_path)
    image1 = Image.open(img_path).convert('RGB')
    image1 = image1.resize((224, 224))
    prep_img = preprocess_image(image1, resize_im=(224, 224))
    print(model)
    a, output = model(prep_img)
    print(output)
    target = output.argmax() + 1
    target = target.item()
    file_name_to_export = './{}_target{}_layer{}_class{}.jpg'.format(finename, target, target_layer, target_class)
    # Grad cam
    grad_cam = GradCam(model, target_layer=target_layer)
    # Generate cam mask
    cam = grad_cam.generate_cam(prep_img, target_class=target_class)
    # Save mask
    save_class_activation_images(image1, cam, file_name_to_export)
    print('Grad cam completed')


def run():
    from models import zsl_att
    model_path = '/home/seven/workspace/example/asset/main_class_att_raf_rafdb_v2_0.2_bs64_202010071947/ckpt/best'
    model = zsl_att.zsl_resnet18(num_classes=7, attribute_dim=28).to('cpu')
    model = load_model(model, model_path)

    img_path = '/home/seven/datasets/RAF/basic/Image/aligned/train_04602_aligned.jpg'
    get_cam(model, target_layer=7, img_path=img_path, target_class=1)


if __name__ == '__main__':
    run()
    # # Get params
    #
    # from torch.autograd import Variable
    # from models import zsl_att
    #
    # # device = torch.device("cpu")
    # img_path = '/home/seven/datasets/palm2/train/2/01142115249648_2.jpg'
    # target_class = 1
    #
    # model_path = '/home/seven/workspace/example/asset/main_class_att_raf_rafdb_v2_0.2_bs64_202010071947/ckpt/best'
    # model = zsl_att.zsl_resnet18(num_classes=7, attribute_dim=28).to('cpu')
    # model = load_model(model, model_path)
    #
    # image1 = Image.open(img_path).convert('RGB')
    # image1 = image1.resize((224, 224))
    # prep_img = preprocess_image(image1, resize_im=(224, 224))
    # print(model)
    # output = model(prep_img)
    # print(output)
    # target = output.argmax() + 1
    # target = target.item()
    # file_name_to_export = './cam7_2_{}_01142115249648_2.jpg'.format(target)
    # # Grad cam
    # grad_cam = GradCam(model, target_layer=7)
    # # Generate cam mask
    # cam = grad_cam.generate_cam(prep_img, target_class=target_class)
    # # Save mask
    # save_class_activation_images(image1, cam, file_name_to_export)
    # print('Grad cam completed')
