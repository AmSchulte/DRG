import streamlit as st
import matplotlib.pyplot as plt
import os
import numpy as np
from skimage.color import hsv2rgb


d7 = '/app/drg/examples/rats d7'    
d14 = '/app/drg/examples/rats d14'

condition = st.sidebar.selectbox('Condition', ["SNI", "sham"])

time = st.sidebar.selectbox('Days after OP', ["7 days", "14 days"])

rats = 5 if condition == "sham" and time == '14 days' else 6

rat = st.sidebar.slider('Rat number', min_value=1, value=1, max_value=rats)
drg_position = st.sidebar.selectbox('DRG position', ["L4IL", "L4CL", "L5IL", "L5CL"])
image_type = st.sidebar.selectbox('Staining', ["NF", "GS", "GFAP"])

#test = st.sidebar.button('test')

if time == '7 days':
    path = d7
elif time == '14 days':
    path = d14
else:
    path = d7

if rat == 1 and time == '7 days':
    image_folder_path = os.path.join(path, condition, "rat "+str(rat), drg_position, image_type)
    filenames = os.listdir(image_folder_path)

    image_number = st.sidebar.slider('Image number', min_value=1, value=1, max_value=len(filenames))

    img_path = os.path.join(image_folder_path, '{0:04d}'.format(image_number)+'.tif') 
    mask_folder_path = image_folder_path + '_pred/masks'
    mask_path = os.path.join(mask_folder_path, '{0:04d}'.format(image_number)+'.png')

    st.title('rat DRG image segmentation')

    st.header(image_type)

    img = plt.imread(img_path)[::2, ::2]
    mask = plt.imread(mask_path)[::2, ::2]

    left, right = st.sidebar.columns(2)
    show_mask = left.checkbox("Mask")
    show_separate_mask = right.checkbox("separate mask")


    gain = st.sidebar.slider('Gain', 0.0, 3.0, 1.0)
    bias = st.sidebar.slider('Bias (brightness)', 0.0, 0.5, 0.1)
    img = (img  - np.min(img ))/np.ptp(img )
    img = np.clip(img * gain + bias, 0, 1)

    mask_template = np.zeros([img.shape[0], img.shape[1], 3])
    mask_template[..., 2] = img
    mask_template[..., 1] = mask/2
    mask_template[..., 0] = 0.5
    mask_final = hsv2rgb(mask_template)

    if show_mask:
        st.image(image=mask_final, width=500)
    elif show_separate_mask:
        st.image(image=mask, width=500)
    else:
        st.image(image=img, width=500)




    #st.image(image=mask)
    #st.pyplot(fig)

    '''
    Image segmentation was performed with Deepflash2, an open source Deep-Learning tool for biological image segmentation
    "https://matjesg.github.io/deepflash2/"
    '''

    '''
    For visual purposes, original images were normalized by the min-max values of the respective image
    '''
else:
    st.write("Data not shown.\n For full dataset, see: ")


