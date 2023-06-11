from segmenter import segment_image, label_lookup_dict
from diffuser import diffuse_segmented_img
from utilities import show_all_results
from PIL import Image


def main(img_path: str,
         label: str,
         prompt: str,
         seed: int=None,
         negative_prompt: str=None,
         num_inference_steps: int=100,
         verbose: bool=False)->None:

    ## input validation
    # check for required arguments
    if img_path is None:
        print('FAILURE: must enter img_path')
        
    # check if img_path leads to a valid image
    try:
        img = Image.open(img_path)
    except:
        print('FAILURE: img_path leads to invalid image')
        return
    
    # check that label is contained in label_lookup_dict keys 
    if label not in label_lookup_dict.keys():
        print('FAILURE: label not found in label_lookup_dict')
        return
    
    ## perform segmentation
    # if verbose print update
    if verbose:
        print('performing segmentation...')
    labels = [label]
    img, mask, seg = segment_image(img_path=img_path,
                                   labels=labels)
    if verbose:
        print('segmentation complete.')

    ## diffuse the masked segmentation 
    if verbose:
        print('diffusing segmentation...')
    diffused_img = diffuse_segmented_img(img=img,
                                         mask=mask,
                                         prompt=prompt,
                                         seed=seed)
    if verbose:
        print('diffusion complete.')

    # show results
    show_all_results(seg.orig_img,
                     mask,
                     diffused_img)
