import torch
import torch.nn as nn
from torchvision.models.vgg import vgg16

dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

def bce_loss(input, target):
    """
    Numerically stable version of the binary cross-entropy loss function in PyTorch.

    Inputs:
    - input: PyTorch Tensor of shape (N, ) giving scores.
    - target: PyTorch Tensor of shape (N,) containing 0 and 1 giving targets.

    Returns:
    - A PyTorch Tensor containing the mean BCE loss over the minibatch of input data.
    """
    bce = nn.BCEWithLogitsLoss()
    return bce(input.squeeze(), target)

def discriminator_loss(logits_real, logits_fake):
    """
    Computes the discriminator loss described above.

    Inputs:
    - logits_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Returns:
    - loss: PyTorch Tensor containing (scalar) the loss for the discriminator.
    """
    loss = None
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    target_fake = torch.zeros(logits_fake.shape).type(dtype)
    target_real = torch.ones(logits_real.shape).type(dtype)
    loss = bce_loss(logits_real, target_real.squeeze()) + bce_loss(logits_fake, target_fake.squeeze())

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return loss
  

class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        vgg = vgg16(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()
        self.tv_loss = TVLoss()

    def forward(self, logits_fake, out_images, target_images):
        # Adversarial Loss
        target_fake = torch.ones(logits_fake.shape).type(dtype)
        adversarial_loss = bce_loss(logits_fake, target_fake.squeeze())
        # Perception Loss
        out_feat = self.loss_network(out_images.repeat([1,3,1,1]))
        target_feat = self.loss_network(target_images.repeat([1,3,1,1]))
        perception_loss = self.mse_loss(out_feat.view(out_feat.size(0),-1), target_feat.view(target_feat.size(0),-1))
        # Image Loss
        image_loss = self.mse_loss(out_images, target_images)
        # TV loss
        tv_loss = self.tv_loss(out_images)

        return image_loss + 0.001 * adversarial_loss + 0.006 * perception_loss + 2e-8 * tv_loss
        #return adversarial_loss

class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        b, c, h, w = x.shape
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h-1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w-1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / b

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]
