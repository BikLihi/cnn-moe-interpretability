"""
    @author: Utku Ozbulak 
    source: https://github.com/utkuozbulak/pytorch-cnn-visualizations

    Some modifications have been made
"""

import os
import numpy as np
import copy
import cv2
from PIL import Image, ImageFilter
import matplotlib.cm as mpl_color_map
import torch
from torch.autograd import Variable
from torch.optim import Adam


class CNNLayerVisualization():
    """
        Produces an image that minimizes the loss of a convolution
        operation for a specific layer and filter
    """
    def __init__(self, model, selected_layer, selected_filter, mean, std, save_path=None):
        self.model = model
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.selected_layer = selected_layer
        self.selected_filter = selected_filter
        self.conv_output = 0
        self.save_path = save_path
        self.mean = mean
        self.std = std
    
    def preprocess_image(self, pil_im, resize_im=False):
        """
            Processes image for CNNs
        Args:
            PIL_img (PIL_img): PIL Image or numpy array to process
            resize_im (bool): Resize to 224 or not
        returns:
            im_as_var (torch variable): Variable that contains processed float tensor
        """
        #ensure or transform incoming image to PIL image
        if type(pil_im) != Image.Image:
            try:
                pil_im = Image.fromarray(pil_im)
            except Exception as e:
                print("could not transform PIL_img to a PIL Image object. Please check input.")

        # Resize image
        if resize_im:
            pil_im = pil_im.resize((224, 224), Image.ANTIALIAS)

        im_as_arr = np.float32(pil_im)
        im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
        # Normalize the channels
        for channel, _ in enumerate(im_as_arr):
            im_as_arr[channel] /= 255
            im_as_arr[channel] -= self.mean[channel]
            im_as_arr[channel] /= self.std[channel]
        # Convert to float tensor
        im_as_ten = torch.from_numpy(im_as_arr).float()
        # Add one more channel to the beginning. Tensor shape = 1,3,224,224
        im_as_ten.unsqueeze_(0)
        # Convert to Pytorch variable
        im_as_var = Variable(im_as_ten, requires_grad=True)
        return im_as_var

    def recreate_image(self, im_as_var):
        """
            Recreates images from a torch variable, sort of reverse preprocessing
        Args:
            im_as_var (torch variable): Image to recreate
        returns:
            recreated_im (numpy arr): Recreated image in array
        """
        reverse_mean = np.array(self.mean) * (-1)
        reverse_std = 1 / np.array(self.std)
        recreated_im = copy.copy(im_as_var.data.numpy()[0])
        for c in range(3):
            recreated_im[c] /= reverse_std[c]
            recreated_im[c] -= reverse_mean[c]
        recreated_im[recreated_im > 1] = 1
        recreated_im[recreated_im < 0] = 0
        recreated_im = np.round(recreated_im * 255)

        recreated_im = np.uint8(recreated_im).transpose(1, 2, 0)
        return recreated_im

    def save_image(self, im, path):
        """
            Saves a numpy matrix or PIL image as an image
        Args:
            im_as_arr (Numpy array): Matrix of shape DxWxH
            path (str): Path to the image
        """
        if isinstance(im, (np.ndarray, np.generic)):
            im = self.format_np_output(im)
            im = Image.fromarray(im)
        im.save(path)
    
    def format_np_output(self, np_arr):
        """
        This is a (kind of) bandaid fix to streamline saving procedure.
        It converts all the outputs to the same format which is 3xWxH
        with using sucecssive if clauses.
        Args:
            im_as_arr (Numpy array): Matrix of shape 1xWxH or WxH or 3xWxH
        """
        # Phase/Case 1: The np arr only has 2 dimensions
        # Result: Add a dimension at the beginning
        if len(np_arr.shape) == 2:
            np_arr = np.expand_dims(np_arr, axis=0)
        # Phase/Case 2: Np arr has only 1 channel (assuming first dim is channel)
        # Result: Repeat first channel and convert 1xWxH to 3xWxH
        if np_arr.shape[0] == 1:
            np_arr = np.repeat(np_arr, 3, axis=0)
        # Phase/Case 3: Np arr is of shape 3xWxH
        # Result: Convert it to WxHx3 in order to make it saveable by PIL
        if np_arr.shape[0] == 3:
            np_arr = np_arr.transpose(1, 2, 0)
        # Phase/Case 4: NP arr is normalized between 0-1
        # Result: Multiply with 255 and change type to make it saveable by PIL
        if np.max(np_arr) <= 1:
            np_arr = (np_arr*255).astype(np.uint8)
        return np_arr


    def hook_layer(self):
        def hook_function(module, grad_in, grad_out):
            # Gets the conv output of the selected filter (from selected layer)
            #grad_out = torch.tensor(grad_out, requires_grad=True).cuda()
            if type(grad_out) is dict:
                self.conv_output = grad_out['output'][0, self.selected_filter]
            else:
                self.conv_output = grad_out[0, self.selected_filter]
        # Hook the selected layer
        return self.selected_layer.register_forward_hook(hook_function)
    
    
    def visualise_layer_with_hooks(self, init_size=8, scaling_factor=1.2, scaling_steps=25, iterations_per_level=10, lr=0.1, file_name=None, verbose=False):
        # Hook the selected layer
        handle = self.hook_layer()
        # Generate a random image
        np.random.seed(42)
        random_image = np.uint8(np.random.uniform(150, 180, (init_size, init_size, 3)))
        # Process image and return variable
        processed_image = self.preprocess_image(random_image, False)
        for i in range(scaling_steps):
            optimizer = Adam([processed_image], lr=lr, weight_decay=1e-6)
            for j in range(iterations_per_level):
                optimizer.zero_grad()
                # Assign create image to a variable to move forward in the model
                x = processed_image.cuda()
                for param in self.model.parameters():
                    param.requires_grad = False

                self.model(x)
                loss = -torch.mean(self.conv_output)
                if verbose:
                    print('Iteration:', str(iterations_per_level * i + j), 'Loss:', "{0:.2f}".format(loss.data.cpu().numpy()))
                # Backward
                loss.backward()
                # Update image
                optimizer.step()

            # Recreate image
            self.created_image = self.recreate_image(processed_image)
            # Save image
            img = processed_image.data.cpu().numpy()[0].transpose(1,2,0)
            sz = int(scaling_factor * processed_image.shape[2])  # calculate new image size
            img = cv2.resize(img, (sz, sz), interpolation = cv2.INTER_CUBIC)   # scale image up
            img = cv2.blur(img,(1, 1))  # blur image to reduce high frequency patterns
            img = img.transpose(2,0,1)
            processed_image = torch.tensor(img[None], requires_grad=True, dtype=torch.float)  # convert image to Variable that requires grad
        if self.save_path:
            if file_name:
                self.save_image(self.created_image, self.save_path + file_name + '_f' + str(self.selected_filter) + '.png')
            else:
                im_path = self.save_path + str(self.selected_layer) + \
                    '_f' + str(self.selected_filter) + '.png'
                self.save_image(self.created_image, im_path)
        handle.remove()
        return self.created_image

    def visualise_mean_activations(self, image, file_name=None):
        # Hook the selected layer
        self.hook_layer()
        # Generate a random image

        processed_image = self.preprocess_image(random_image, False)
        for i in range(25):
            lr = schedule(i)
            optimizer = Adam([processed_image], lr=lr, weight_decay=1e-6)
            for j in range(10):
                optimizer.zero_grad()
                # Assign create image to a variable to move forward in the model
                x = processed_image.cuda()
                self.model.set_parameter_requires_grad(False)
                self.model(x)
                loss = -torch.mean(self.conv_output)
                print('Iteration:', str(10 * i + j), 'Loss:', "{0:.2f}".format(loss.data.cpu().numpy()))
                # Backward
                loss.backward()
                # Update image
                optimizer.step()

            # Recreate image
            self.created_image = self.recreate_image(processed_image)
            # Save image
            img = processed_image.data.cpu().numpy()[0].transpose(1,2,0)
            sz = int(scaling_factor * processed_image.shape[2])  # calculate new image size
            img = cv2.resize(img, (sz, sz), interpolation = cv2.INTER_CUBIC)   # scale image up
            img = cv2.blur(img,(1, 1))  # blur image to reduce high frequency patterns
            img = img.transpose(2,0,1)
            processed_image = torch.tensor(img[None], requires_grad=True, dtype=torch.float)  # convert image to Variable that requires grad
        if file_name:
            im_path = self.save_path + file_name + '_' + str(self.selected_layer) + \
            '_f' + str(self.selected_filter) + '.png'
        else:
            im_path = self.save_path + str(self.selected_layer) + \
                '_f' + str(self.selected_filter) + '.png'
        self.save_image(self.created_image, im_path)
