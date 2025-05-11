Developed a Vision Transformer (ViT) from scratch for sequence prediction of non-interacting spherical particles in a box. 
This model processes input frames by dividing them into patches and uses a class token from the encoder's output to predict the next frame. 
Successfully achieved accurate predictions up to 1 frame ahead, demonstrating the ViT's effectiveness in modeling particle dynamics.

1. Each frame is converted into a latent variable of dimension 2. This is done in two ways 
  1. Autoencoder (Autoencoder.ipynb)
  2. Vision transformer (Vision_transformer.ipynb)
2. These latent variables are used as input for transformers. Transformers are used to predict the 6th frame given 5 initial frames. Transformer code  and reuslts are in Vision_transformer.ipynb 
