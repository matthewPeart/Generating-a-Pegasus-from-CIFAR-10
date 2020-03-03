# Generating a Pegasus from CIFAR-10

In this project we train a Deep Convolutional Generative Adversarial Network (DCGAN) [1] to produce a pegasus from the horse and bird images of the CIFAR-10 [2] dataset. We train the DCGAN on both the horse and bird images in an attempt to learn a joint latent distribution. We then interpolate the latent space to find our picture of a pegasus. The code is written in a Python notebook for easy experimentation on Google Colab. Below outlines the methodology, results, and findings of the project.

## Deep Convolutional Generative Adversarial Networks:

A DCGAN [1] is an architecture composed of two models; a generator and a discriminator. The models are trained in a min-max game, where the generator learns how to convert vectors sampled from a prior into real images, and the discriminator learns how to discriminate the generated examples (fake) from the real images. The models are trained in tandem:

* Train the generator network:
  1. Generate a mini-batch of fake images by passing noise vectors sampled from the prior through the generator.
  2. Measure the loss on how well the generator fools the discriminator, backpropagate the error and step the optimiser.

* Train the discriminator network:
  1. Generate a mini-batch of fake images by passing noise vectors sampled from the prior through the generator.
  2. Measure the discriminator loss on both real and fake images, backpropagate the error and step the optimiser.

<p align="center">
<img src="/figures/gan.png" height="55%" width="55%">
</p>

## Architecture:

We follow most of the guidance outlined in [1]:

1. Replace max pooling layers with striding convolutions, allowing the network to learn its own spatial sampling.

2. Eliminate fully connected hidden layers in both networks to improve model stability.

3. Use batch normalisation to stabilize learning, to help the gradient flow, and to prevent the network from collapsing.

Unlike in the paper, the LeakyReLU activation function was used in the generator network as well as the discriminator network for all layers. The output of the generator network uses the Tanh function, and the output of the discriminator network uses a sigmoid function to keep the predictions squished between 0 and 1. The final architecture can be seen in the diagram below.

<p align="center">
<img src="/figures/architectures.png" height="55%" width="55%">
</p>

A normal distribution with mean = 0 and variance = 1 was used for the prior. It was found that mini-batches of size 16 kept training stable. The generator loss is measured as the binary cross entropy of the predictions made by the discriminator network on the fake images. To measure the loss of the discriminator, both real and fake images are passed through the network and an average of the binary cross entropy loss is taken. Batch normalization helped to keep training stable.

## Training:

The figure below shows the loss plots of the generator and discriminator networks. The losses fluctuate as the generator learns the distribution of horse and bird images, and as the discriminator learns how to accurately separate the real and fake images. After the 300th epoch, the discriminator loss steeply dropped as the network learnt how to spot fakes. The generator network never managed to catch back up; this is called non-convergence. Several runs also resulted in mode collapse.

<p align="center">
<img src="/figures/training.png" height="75%" width="75%">
</p>

## Results:

The figure below shows sixteen 32x32 generated images after the last epoch. The generator has learnt horse-like shapes, and in some cases bird-like wings. To find an image of a pegasus, the joint latent distribution of the horses and birds was searched by sampling a range of vectors from the normal distribution. The top right image vaguely looks like a horse with wings.

<p align="center">
<img src="/figures/results.png" height="45%" width="45%">
</p>

## Conclusions and Recommendations:

1. Deep Convolutional Generative Adversarial Networks can learn representations and generate novel images.

2. It is difficult to train DCGANs on small images (32x32) as it can result in image blur.

3. It may have been better to train two DCGANs on horses and birds separately, and then interpolate the images.

4. Regularisation methodology could be tweaked to improve convergence between the two models. 

## References:

[1] Radford, A., Metz, L. and Chintala, S., 2015. Unsupervised representation learning with deep convolutional generative adversarial networks. arXiv preprint arXiv:1511.06434.

[2] Alex Krizhevsky, 2009. Learning Multiple Layers of Features from Tiny Images. https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf.
