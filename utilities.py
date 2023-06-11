import urllib.request
from PIL import Image
import matplotlib.pyplot as plt

def download_file(url, output_path):
    urllib.request.urlretrieve(url, output_path)
    
def show_img(image_path):
    # Open the image file
    image = Image.open(image_path)

    # Display the image
    plt.imshow(image)
    plt.axis('off')  
    plt.show()

def show_all_results(segmented_img,
                     mask,
                     diffused_img):
    # Create a figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(10, 5))

    # Display the first image
    # axes[0].imshow(cv2.cvtColor(segmented_img, cv2.COLOR_BGR2RGB))
    axes[0].imshow(segmented_img)

    axes[0].axis('off')
    axes[0].set_title('segmented image')

    # Display the second image
    axes[1].imshow(mask)
    axes[1].axis('off')
    axes[1].set_title('object segmentation')

    # Display the third image
    axes[2].imshow(diffused_img)
    axes[2].axis('off')
    axes[2].set_title('diffused image')

    # Adjust the layout
    plt.tight_layout()

    # Show the figure
    plt.show()




