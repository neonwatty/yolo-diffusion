from PIL import Image
import cv2
import torch
from diffusers import StableDiffusionInpaintPipeline

# define diffusion pipeline - first call will download model
device = "cuda"
diffusion_pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
     torch_dtype=torch.float16).to(device)

# turn off progress bars when processing input image
# diffusion_pipe.set_progress_bar_config(leave=False)
# diffusion_pipe.set_progress_bar_config(disable=True)


# diffuse image / mask
def diffuse_segmented_img(img,
                          mask,
                          prompt="realistic coca cola can, high resolution, high quality, logo visible, red and white colors, aluminum",
                          negative_prompt='fingers, dented can, crushed can, bottle, plastic, low quality, low resolution',
                          seed=None,
                          num_inference_steps=100
                        ):

    # prep img for diffusion
    img_pil = Image.fromarray(img).convert('RGB').resize((512, 512))

    # prep mask for diffusion
    mask_pil = Image.fromarray(mask).convert('RGB').resize((512, 512))

    # difuse mask
    if seed is not None:
        if isinstance(seed,int):
            torch.manual_seed(seed)
    else:
        print(f'your seed is {torch.seed()}')
    diffused_img = diffusion_pipe(
                                  prompt=prompt,
                                  negative_prompt=negative_prompt,
                                  image=img_pil,
                                  mask_image=mask_pil,
                                  num_inference_steps=num_inference_steps,
                                  output_type='np.array').images[0]

    # resize to match input img / mask 
    diffused_img = cv2.resize(diffused_img, (img.shape[1], img.shape[0]))
    return diffused_img
