![Generative Models](https://img.shields.io/badge/Generative_Models-0096FA?style=flat-square&logo=pixiv&logoColor=0096FA&label=Tag&labelColor=212030)
![Techs](https://img.shields.io/badge/Pytorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=EE4C2C&label=Tech&labelColor=212030)
![Status](https://img.shields.io/badge/Working-19947C?style=flat-square&logo=esbuild&logoColor=24DBB6&label=Status&labelColor=212030)

![Main Image](images/epoch_10.gif)
## About
This personal project synthesize animal images via **Conditional-GAN**, extended generative adversarial network.

During the training process, the label is additionally given to the generator as input and the discriminator determines its class. Then, we can manipulate the generator's input during the sampling process to get the desired class.

The project's goal is to check whether we can sample the synthesized animal images by giving **multi-labeled input** to the generator.
## Dataset
### Animal Faces-HQ(AFHQ)
Introduced by Choi et al. in `StarGAN v2: Diverse Image Synthesis for Multiple Domains`.

AFHQ is a dataset of animal faces consisting of 15,000 high-quality images at 512 × 512 resolution.  
The dataset includes three domains of cat, dog, and wildlife, each providing 5000 images.  
All images are vertically and horizontally aligned to have the eyes at the center.
### Download dataset
```sh
kaggle datasets download -d andrewmvd/animal-faces -p ./dataset/ --unzip
```
### Dataset Directory
```
dataset
|-- afhq
|   |-- train
|   |   |-- cat
|   |   |   |-- flickr_cat_000002.jpg
|   |   |   |-- ...
|   |   |-- dog
|   |   |-- wild
|   |-- val
|   |   |-- cat
|   |   |-- ...
```
## Model Architecture
*TBU*  
## Results
### Epochs: 10
![Image 10](images/epoch_10.gif)
### Epochs: 100
![Image 100](images/epoch_100.png)
### Epochs: 500
<table align="center">
<tr>
    <th>Cat</th>
    <th>Dog</th>
    <th>Catdog</th>
</tr>
<tr>
    <td><img src="images/epoch_500_cat.png" width="100"/></td>
    <td><img src="images/epoch_500_dog.png" width="100"/></td>
    <td><img src="images/epoch_500_catdog.png" height="100"/></td>
</tr>
</table>

## TODO
* Reconstruct overall architecture
* .ipynb -> .py
