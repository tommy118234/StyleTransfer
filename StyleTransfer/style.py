# system imports
import os

# library imports
import caffe
import cv2
import numpy as np
from scipy.linalg.blas import sgemm
from scipy.optimize import minimize
from skimage.transform import rescale

# numeric constants
INF = np.float32(np.inf)
STYLE_SCALE = 1.2


def _compute_style_grad(F, G, G_style, layer):
    """
        Computes style gradient and loss from activation features.
       """
    # compute loss and gradient
    (Fl, Gl) = (F[layer], G[layer])
    c = Fl.shape[0]**-2 * Fl.shape[1]**-2
    El = Gl - G_style[layer]
    loss = c/4 * (El**2).sum()
    grad = c * sgemm(1.0, El, Fl) * (Fl > 0)
    return loss, grad 


def _compute_content_grad(F, F_content, layer):
    """
        Computes content gradient and loss from activation features.
       """
    # compute loss and gradient
    """
    TODO #7
    Compute gradient and loss for content
    """
    (Fl, El) = (F[layer], F[layer] - F_content[layer])
    loss = (El ** 2).sum() / 2
    grad = El * (Fl > 0)
    return loss, grad


def _compute_reprs(net_in, net, layers_style, layers_content, gram_scale=1):
    """
        Computes representation matrices for an image.
        :param net_in: content image or style image
        :param net: caffe network
        :param layers_style: layers selected for Style Target. 
            If net_in is content image, this should be []
        :param layers_content: layers selected for Content Target. 
            If net_in is style image, this should be []
       """
    # input data and forward pass
    (repr_s, repr_c) = ({}, {})
    net.blobs["data"].data[0] = net_in
    net.forward()
    # decide if net_in is content image or style image
    if layers_style == []:
        repr_s = {}
    if layers_content == []:
        repr_c = {}
    """
    TODO #6
    Calculate representations for content and style
    """
    for layer in (set(layers_style) | set(layers_content)):
        f_sc = net.blobs[layer].data[0].copy()
        (a, b, c) = f_sc.shape
        f_sc = np.reshape(f_sc, (a, b * c))
        repr_c[layer] = f_sc
        if layer in layers_style:
            repr_s[layer] = sgemm(gram_scale, f_sc, f_sc.T)
    return repr_s, repr_c


def style_optfn(x, net, weights, layers, reprs, ratio):
    """
        Style transfer optimization callback for scipy.optimize.minimize().
        :param numpy.ndarray x:
            Flattened data array.
        :param caffe.Net net:
            Network to use to generate gradients.
        :param dict weights:
            Weights to use in the network.
        :param list layers:
            Layers to use in the network.
        :param tuple reprs:
            Representation matrices packed in a tuple.
        :param float ratio:
            Style-to-content ratio.
        """
    # update params
    layers_style = weights["style"].keys()
    layers_content = weights["content"].keys()
    net_in = x.reshape(net.blobs["data"].data.shape[1:])
    # compute representations
    (G_style, F_content) = reprs
    (G, F) = _compute_reprs(net_in, net, layers_style, layers_content)
    # backprop by layer
    """
    TODO #8
    Compute gradient and loss for the objective function
    """
    loss = 0
    net.blobs[layers[-1]].diff[:] = 0
    for i, layer in enumerate(reversed(layers)):
        next_layer = None if i == len(layers)-1 else layers[-i-2]
        grad = net.blobs[layer].diff[0]
        # style loss
        if layer in layers_style:
            wl = weights["style"][layer]
            (l, g) = _compute_style_grad(F, G, G_style, layer)
            loss += wl * l * ratio
            grad += wl * g.reshape(grad.shape) * ratio
        # content loss
        if layer in layers_content:
            wl = weights["content"][layer]
            (l, g) = _compute_content_grad(F, F_content, layer)
            loss += wl * l
            grad += wl * g.reshape(grad.shape)
        # compute final gradient
        net.backward(start=layer, end=next_layer)
        if next_layer is None:
            grad = net.blobs["data"].diff[0]
        else:
            grad = net.blobs[next_layer].diff[0]
    # format gradient for minimize() function
    grad = grad.flatten().astype(np.float64)
    return loss, grad


class StyleTransfer(object):
    """
        Style transfer class.
       """
    def __init__(self):
        """
            Initialize the model used for style transfer.
             """
        # path configuration
        style_path = os.path.abspath(os.path.split(__file__)[0])
        base_path = os.path.join(style_path, "models/vgg16")

        model_file = os.path.join(base_path, "VGG_ILSVRC_16_layers_deploy.prototxt")
        pretrained_file = os.path.join(base_path, "VGG_ILSVRC_16_layers.caffemodel")
        mean_file = os.path.join(base_path, "ilsvrc_2012_mean.npy")
        weights = {"content": {"conv4_2": 1},
                   "style": {"conv1_1": 0.2,
                             "conv2_1": 0.2,
                             "conv3_1": 0.2,
                             "conv4_1": 0.2,
                             "conv5_1": 0.2}}

        # add model and weights
        self.grad_iter = 0
        self.load_model(model_file, pretrained_file, mean_file)
        self.weights = weights.copy()
        self.layers = []
        for layer in self.net.blobs:
            if layer in self.weights["style"] or layer in self.weights["content"]:
                self.layers.append(layer)

    def load_model(self, model_file, pretrained_file, mean_file):
        """
            Loads specified model from caffe install (see caffe docs).

            :param str model_file:
                Path to model protobuf.

            :param str pretrained_file:
                Path to pretrained caffe model.

            :param str mean_file:
                Path to mean file.
              """
        # load net (supressing stderr output)
        null_fds = os.open(os.devnull, os.O_RDWR)
        out_orig = os.dup(2)
        os.dup2(null_fds, 2)
        net = caffe.Net(model_file, pretrained_file, caffe.TEST)
        os.dup2(out_orig, 2)
        os.close(null_fds)
        # all models used are trained on imagenet data
        transformer = caffe.io.Transformer({"data": net.blobs["data"].data.shape})
        transformer.set_mean("data", np.load(mean_file).mean(1).mean(1))
        transformer.set_channel_swap("data", (2,1,0))
        transformer.set_transpose("data", (2,0,1))
        transformer.set_raw_scale("data", 255)
        # add net parameters
        self.net = net
        self.transformer = transformer

    def get_generated(self):
        """
             Saves the generated image (net input, after optimization).

             :param str path:
                Output path.
             """
        data = self.net.blobs["data"].data
        img_out = self.transformer.deprocess("data", data)
        return img_out

    def _rescale_net(self, img):
        """
              Rescales the network to fit a particular image.
             """
        # get new dimensions and rescale net + transformer
        new_dims = (1, img.shape[2]) + img.shape[:2]
        self.net.blobs["data"].reshape(*new_dims)
        self.transformer.inputs["data"] = new_dims

    def transfer_style(self, img_style, img_content, length=512, ratio=1e5,
                       n_iter=512, init="-1", verbose=False):
        """
            Transfers the style of the artwork to the input image.

            :param numpy.ndarray img_style:
                A style image with the desired target style.

            :param numpy.ndarray img_content:
                A content image in floating point, RGB format.
        """
        # assume that convnet input is square
        orig_dim = min(self.net.blobs["data"].shape[2:])
        # rescale the images
        scale = max(length / float(max(img_style.shape[:2])),
                    orig_dim / float(min(img_style.shape[:2])))
        img_style = rescale(img_style, STYLE_SCALE*scale)
        scale = max(length / float(max(img_content.shape[:2])),
                    orig_dim / float(min(img_content.shape[:2])))
        img_content = rescale(img_content, scale)
        # compute style representations
        self._rescale_net(img_style)
        layers = self.weights["style"].keys()
        net_in = self.transformer.preprocess("data", img_style)
        gram_scale = float(img_content.size)/img_style.size
        G_style = _compute_reprs(net_in, self.net, layers, [],
                                 gram_scale=1)[0]
        # compute content representations
        self._rescale_net(img_content)
        layers = self.weights["content"].keys()
        net_in = self.transformer.preprocess("data", img_content)
        F_content = _compute_reprs(net_in, self.net, [], layers)[1]
        # generate initial net input
        # "content" = content image, see kaishengtai/neuralart
        img0 = self.transformer.preprocess("data", img_content)
        # compute data bounds
        data_min = -self.transformer.mean["data"][:,0,0]
        data_max = data_min + self.transformer.raw_scale["data"]
        data_bounds = [(data_min[0], data_max[0])]*(img0.size/3) + \
                      [(data_min[1], data_max[1])]*(img0.size/3) + \
                      [(data_min[2], data_max[2])]*(img0.size/3)
        # optimization params
        grad_method = "L-BFGS-B"
        reprs = (G_style, F_content)
        minfn_args = {
            "args": (self.net, self.weights, self.layers, reprs, ratio),
            "method": grad_method, "jac": True, "bounds": data_bounds,
            "options": {"maxcor": 8, "maxiter": n_iter, "disp": verbose}
        }
        # optimize
                # set the callback function
        def callback(xk):
            self.grad_iter += 1
            print("Iteration: " + str(self.grad_iter) + '/' + str(n_iter))
        minfn_args["callback"] = callback
        res = minimize(style_optfn, img0.flatten(), **minfn_args).nit
        return res