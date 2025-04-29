import streamlit as st
import matplotlib.pyplot as plt
import os
import numpy as np
from skimage.color import hsv2rgb


path = 'examples/rats d7'    

condition = st.sidebar.selectbox('Condition', ["SNI", "sham"])


drg_position = st.sidebar.selectbox('DRG position', ["L4IL", "L4CL", "L5IL", "L5CL"])
image_type = st.sidebar.selectbox('Staining', ["NF", "GS", "GFAP"])


image_folder_path = os.path.join(path, condition, "rat 1", drg_position, image_type)
filenames = os.listdir(image_folder_path)

image_number = st.sidebar.slider('Image number', min_value=1, value=1, max_value=len(filenames))

img_path = os.path.join(image_folder_path, '{0:04d}'.format(image_number)+'.tif') 
mask_folder_path = image_folder_path + '_pred/masks'
mask_path = os.path.join(mask_folder_path, '{0:04d}'.format(image_number)+'.png')

st.title('rat DRG image segmentation')

st.header(image_type)

@st.cache_data(suppress_st_warning=True)
def read_images(img_path, mask_path):
    img = plt.imread(img_path)[::2, ::2]
    mask = plt.imread(mask_path)[::2, ::2]
    return img, mask

img, mask = read_images(img_path, mask_path)

left, right = st.sidebar.columns(2)
show_mask = left.checkbox("Mask")
show_separate_mask = right.checkbox("separate mask")

if show_separate_mask:
    show_mask = False


gain = st.sidebar.slider('Gain', 0.0, 3.0, 1.0)
bias = st.sidebar.slider('Bias (brightness)', 0.0, 0.5, 0.1)

@st.cache_data(suppress_st_warning=True)
def scale_images(img, mask):
    img = (img  - np.min(img ))/np.ptp(img )
    img = np.clip(img * gain + bias, 0, 1)

    mask_template = np.zeros([img.shape[0], img.shape[1], 3])
    mask_template[..., 2] = img
    mask_template[..., 1] = mask/2
    mask_template[..., 0] = 0.5
    mask_final = hsv2rgb(mask_template)
    return img, mask_final

img, mask_final = scale_images(img, mask)

width = round((600/img.shape[0])*img.shape[1])


if show_mask:
    st.image(image=mask_final, width=width)
elif show_separate_mask:
    st.image(image=mask, width=width)
else:
    st.image(image=img, width=width)

'''
Images are from one exemplary rat 7 days after sham or SNI. 
The full dataset is available at: https://doi.org/10.5281/zenodo.6487423
'''

'''
Image segmentation was performed with Deepflash2, an open source Deep-Learning tool for biological image segmentation
"https://matjesg.github.io/deepflash2/"
'''

'''
For visual purposes, original images were normalized by the min-max values of the respective image
'''



