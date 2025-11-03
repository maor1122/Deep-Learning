import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.init import xavier_uniform, xavier_normal


class ConvClassifier(nn.Module):
    """
    A convolutional classifier model based on PyTorch nn.Modules.

    The architecture is:
    [(Conv -> ReLU)*P -> MaxPool]*(N/P) -> (Linear -> ReLU)*M -> Linear
    """
    def __init__(self, in_size, out_classes, filters, pool_every, hidden_dims):
        """
        :param in_size: Size of input images, e.g. (C,H,W).
        :param out_classes: Number of classes to output in the final layer.
        :param filters: A list of length N containing the number of
            filters in each conv layer.
        :param pool_every: P, the number of conv layers before each max-pool.
        :param hidden_dims: List of of length M containing hidden dimensions of
            each Linear layer (not including the output layer).
        """
        super().__init__()
        self.in_size = in_size
        self.out_classes = out_classes
        self.filters = filters
        self.pool_every = pool_every
        self.hidden_dims = hidden_dims

        self.feature_extractor = self._make_feature_extractor()
        self.classifier = self._make_classifier()
        print(self)
    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the feature extractor part of the model:
        # [(Conv -> ReLU)*P -> MaxPool]*(N/P)
        # Use only dimension-preserving 3x3 convolutions (you will need to add padding). Apply 2x2 Max
        # Pooling to reduce dimensions.
        # If P>N you should implement:
        # (Conv -> ReLU)*N
        # Hint: use loop for len(self.filters) and append the layers you need to the list named 'layers'.
        # Use :
        # if <layer index>%self.pool_every==0:
        #     ...
        # in order to append maxpooling layer in the right places.
        # ====== YOUR CODE: ======
        prev_channels = in_channels
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        for i, num_filters in enumerate(self.filters):
            layers.append(nn.Conv2d(in_channels=prev_channels, out_channels=num_filters, kernel_size=3, padding=1).to(self.device))
            layers.append(nn.ReLU(inplace=True).to(self.device))
            prev_channels = num_filters
            if (i + 1) % self.pool_every == 0:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2).to(self.device))
                in_h = (in_h - 1) // 2 + 1
                in_w = (in_w - 1) // 2 + 1
        self.modified_sizes = [prev_channels,in_h,in_w]
        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def _make_classifier(self):
        in_channels, in_h, in_w = tuple(self.in_size)

        layers = []
        # TODO: Create the classifier part of the model:
        # (Linear -> ReLU)*M -> Linear
        # You'll need to calculate the number of features first.
        # The last Linear layer should have an output dimension of out_classes.
        # Hint: use loop for len(self.hidden_dims) and append the layers you need to list named layers.
        # ====== YOUR CODE: ======
        modified_channels, modified_h, modified_w = tuple(self.modified_sizes)
        prev_dim = modified_channels * modified_h * modified_w
        for num_hidden in self.hidden_dims:
          layers.append(nn.Linear(prev_dim, num_hidden).to(self.device))
          layers.append(nn.ReLU(inplace=True).to(self.device))
          prev_dim = num_hidden

        layers.append(nn.Linear(prev_dim, self.out_classes).to(self.device))
        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def forward(self, x):
        # TODO: Implement the forward pass.
        # Extract features from the input (using self.feature_extractor), flatten your result (using torch.flatten),
        # run the classifier on them (using self.classifier) and return class scores.
        # ====== YOUR CODE: ======
        out = self.classifier(torch.flatten(self.feature_extractor(x),1))
        # ========================
        return out


class YourCodeNet(ConvClassifier):
    def __init__(self, in_size, out_classes, filters, pool_every, hidden_dims):
        super().__init__(in_size, out_classes, filters, pool_every, hidden_dims)

    # TODO: Change whatever you want about the ConvClassifier to try to
    # improve it's results on CIFAR-10.
    # For example, add batchnorm, dropout, skip connections, change conv
    # filter sizes etc.
    # ====== YOUR CODE: ======
    
    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # Implement this function with the fixes you suggested question 1.1. Extra points.
        # ====== YOUR CODE: ======
        prev_channels = in_channels
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        for i, num_filters in enumerate(self.filters):
            layers.append(nn.Conv2d(in_channels=prev_channels, out_channels=num_filters, kernel_size=3, padding=1).to(self.device))
            layers.append(nn.BatchNorm2d(num_filters).to(self.device)) #added in ex3 - Batchnormalization
            layers.append(nn.ReLU(inplace=True).to(self.device))
            layers.append(nn.Dropout(p=0.5)) #added in ex4 - Dropout
            prev_channels = num_filters
            if (i + 1) % self.pool_every == 0:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2).to(self.device))
                in_h = (in_h - 1) // 2 + 1
                in_w = (in_w - 1) // 2 + 1
        self.modified_sizes = [prev_channels,in_h,in_w]
        # ========================
        seq = nn.Sequential(*layers)
        return seq

