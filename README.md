# yolo-diffusion

[![collab sticker](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jermwatt/yolo_diffusion/blob/main/object_diffusion_demo.ipynb)


A simple demo for quick experimentation with [Stable Diffusion 2.0](https://huggingface.co/stabilityai/stable-diffusion-2-inpainting) applied to specific objects in 2d images, detected and segmented via [YOLOv8](https://docs.ultralytics.com/).

You can get started in [collab now](https://colab.research.google.com/github/jermwatt/yolo_diffusion/blob/main/object_diffusion_demo.ipynb) - or watch the walkthrough videos below first.



## Overview

With this simple demo you can start with any image - like the one on the left below.

You can then select any object in the image - like the `pizza`.  YOLO will detect and segment this object, and then you can replace it with a diffused suggestion via Stable v2.0.  

![cat cash](https://github.com/jermwatt/yolo_diffusion/blob/main/test_data/cat_cash.png?raw=true)

Like "a pile of cash" - shown on the right.

You can take the same image and do the same thing with another object - like the `cat` - as shown below.

![squirrel pizza](https://github.com/jermwatt/yolo_diffusion/blob/main/test_data/squirrel_pizza.png?raw=true)

Here we used the prompt "'sinister looking cartoon squirrel wearing sunglasses, high resolution'".

Here's another example.  In this case we replace the `person` with 'an ape, smiling, high resolution, holding something"

![squirrel pizza](https://github.com/jermwatt/yolo_diffusion/blob/main/test_data/example_person_replacement.png?raw=true)

YOLOv8 has more than 70 pre-trained objects that can be quickly detected / segmented, then fed to Stable Diffusion.  See the [demo notebook](https://colab.research.google.com/github/jermwatt/yolo_diffusion/blob/main/object_diffusion_demo.ipynb#scrollTo=8GhUUsG2tdZ1) for a complete list of available objects.