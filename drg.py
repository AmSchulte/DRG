import numpy as np
import matplotlib.pyplot as plt
import os
from skimage import feature
from skimage import measure
from skimage import segmentation
from sklearn.preprocessing import RobustScaler
from scipy import misc,ndimage
from tqdm.auto import tqdm
import pandas as pd


class DrgData:
    def __init__(self, subdir):
        nf_path = os.path.join(subdir, 'NF')
        nf_mask_path = os.path.join(subdir, 'NF_pred\masks')
        gs_path = os.path.join(subdir, 'GS')
        gs_mask_path = os.path.join(subdir, 'GS_pred\masks')
        gfap_path = os.path.join(subdir, 'GFAP')
        gfap_mask_path = os.path.join(subdir, 'GFAP_pred\masks')

        self.smallest_roi = self.get_smallest_roi()

        self.neuronal_cell_sizes = []
        self.neuronal_area_per_tissue = []
        self.gs_intensities = []
        self.gfap_intensities = []
        self.gs_intensities_of_area = []
        self.gfap_intensities_of_area = []
        self.gs_rings = []
        self.gfap_rings = []
        self.ring_rings = []
        self.gs_overlaps = []
        self.gfap_overlaps = []
        self.gs_area_per_tissue = []
        self.gfap_area_per_tissue = []
        self.ring_area_per_tissue = []
        self.gs_area_per_neurons = []
        self.gfap_area_per_neurons = []
        self.ring_area_per_neurons = []        

        for nf, nf_mask, gs, gs_mask, gfap, gfap_mask in tqdm(zip(os.listdir(nf_path), os.listdir(nf_mask_path), 
                                                          os.listdir(gs_path), os.listdir(gs_mask_path), 
                                                          os.listdir(gfap_path), os.listdir(gfap_mask_path))):
            
            nf = self.read_img(directory=nf_path, filename=nf)
            nf_mask = self.read_img(directory=nf_mask_path, filename=nf_mask)
            gs = self.read_img(directory=gs_path, filename=gs)
            gs_mask = self.read_img(directory=gs_mask_path, filename=gs_mask)
            gfap = self.read_img(directory=gfap_path, filename=gfap)
            gfap_mask = self.read_img(directory=gfap_mask_path, filename=gfap_mask)

            # Lets count the neurons and their sizes
            nf_label = self.get_nf_label(nf_mask)
            nf_cell_size = self.get_cell_sizes(nf_label)

            # get the neuronal area per tissue area
            # tissue area is defined by a threshold applied on the NF images
            nf_area = nf_mask.sum()
            tissue_mask = (nf>nf.max()*0.05)*1
            tissue_area = tissue_mask.sum()
            if tissue_area == 0:
                nf_area_per_tissue = 0
            else:
                nf_area_per_tissue = nf_area/tissue_area

            # get the GS and GFAP positive area per tissue and neuronal area
            gs_area = gs_mask.sum()
            gfap_area = gfap_mask.sum()
            if gs_area == 0:
                gs_area_tissue = 0
                gs_area_neuron = 0
            elif nf_area == 0 or nf_area <= gs_area:
                gs_area_neuron = 0
                gs_area_tissue = gs_area/tissue_area
            else:
                gs_area_tissue = gs_area/tissue_area
                gs_area_neuron = gs_area/nf_area
            
            if gfap_area == 0:
                gfap_area_tissue = 0
                gfap_area_neuron = 0
            elif nf_area == 0 or nf_area <= gfap_area:
                gfap_area_neuron = 0
                gfap_area_tissue = gfap_area/tissue_area
            else:
                gfap_area_tissue = gfap_area/tissue_area
                gfap_area_neuron = gfap_area/nf_area               
            
 
            # ring area as GS+GFAP positive area
            ring = ((gs_mask+gfap_mask)>0)*1
            ring_area = ring.sum()

            if ring_area == 0:
                ring_area_tissue = 0
                ring_area_neuron = 0
            elif nf_area == 0 or nf_area <= ring_area:
                ring_area_neuron = 0
                ring_area_tissue = ring_area/tissue_area
            else:
                ring_area_tissue = ring_area/tissue_area
                ring_area_neuron = ring_area/nf_area    


            # get the mean intensity per area of GS and GFAP
            gs_norm = self.normalize_image(gs)
            gfap_norm = self.normalize_image(gfap)
            gs_intensity = gs_norm.mean()
            gfap_intensity = gfap_norm.mean()
            gs_intensity_of_area = self.get_intensity_per_area(gs_norm, gs_mask)
            gfap_intensity_of_area = self.get_intensity_per_area(gfap_norm, gfap_mask)

            # calculate overlap of GS and GFAP
            overlap = gs_mask+gfap_mask == 2
            overlap_area = overlap.sum()
            gs_overlap = overlap_area/gs_area
            gfap_overlap = overlap_area/gfap_area

            
            # get the ring size (%) of GS and GFAP around the neurons 
            gs_ring = self.get_rings(nf_label, gs_mask)
            gfap_ring = self.get_rings(nf_label, gfap_mask)
            ring_ring = self.get_rings(nf_label, ring)
            

            # add cell size of neurons, the intensities and ring sizes of GS and GFAP to respective list
            self.neuronal_cell_sizes.append(nf_cell_size)
            self.neuronal_area_per_tissue.append(float(nf_area_per_tissue))
            self.gs_intensities.append(float(gs_intensity))
            self.gfap_intensities.append(float(gfap_intensity))
            self.gs_intensities_of_area.append(float(gs_intensity_of_area))
            self.gfap_intensities_of_area.append(float(gfap_intensity_of_area))
            self.gs_rings.append(gs_ring)
            self.gfap_rings.append(gfap_ring)
            self.ring_rings.append(ring_ring)
            self.gs_overlaps.append(float(gs_overlap))
            self.gfap_overlaps.append(float(gfap_overlap))
            self.gs_area_per_tissue.append(float(gs_area_tissue))
            self.gfap_area_per_tissue.append(float(gfap_area_tissue))
            self.ring_area_per_tissue.append(float(ring_area_tissue))
            self.gs_area_per_neurons.append(float(gs_area_neuron))
            self.gfap_area_per_neurons.append(float(gfap_area_neuron))
            self.ring_area_per_neurons.append(float(ring_area_neuron))        


    def read_img(self, directory, filename):
        path = os.path.join(directory, filename)
        return plt.imread(path)

    def get_cell_sizes(self, label):
        label_flat = np.ndarray.flatten(label)
        colors, counts = np.unique(label_flat, return_counts = True, axis = 0)
        cell_sizes = counts[1:]
        return cell_sizes.tolist()
    
    def normalize_image(self, image):
        shape = image.shape
        transformed = RobustScaler().fit_transform(image.reshape(-1,1))
        image_norm = transformed.reshape(shape)
        # scale image between 0 and 1
        image_norm = (image_norm  - np.min(image_norm))/np.ptp(image_norm)
        return image_norm

    def get_intensity_per_area(self, image_norm, mask):
        masking = mask>0
        masked = image_norm*masking
        return np.sum(masked)/np.sum(mask)

    def get_smallest_roi(self):
        directories = ['F:/Deep Learning/Annotation/NF/experts_annotation/Anne',
                        'F:/Deep Learning/Annotation/NF/experts_annotation/Annemarie',
                        'F:/Deep Learning/Annotation/NF/experts_annotation/johannes']
        smallest_rois = []
        for directory in directories:
            for filename in os.listdir(directory):
                if filename.endswith(".png"):
                    path = os.path.join(directory, filename)
                    mask = plt.imread(path)
                    label = measure.label(mask)
                    unique, counts = np.unique(label, return_counts=True)
                    smallest_roi = np.min(counts)
                    smallest_rois.append(smallest_roi)
        smallest_roi = np.min(smallest_rois)
        return smallest_roi

    def remove_small_rois(self, all_labels):
        unique, counts = np.unique(all_labels, return_counts=True)
        all_labels_dict = dict(zip(unique, counts)) 
           
        list_small_labels = []
        for i in all_labels_dict.keys():
            if all_labels_dict[i] < self.smallest_roi:
                list_small_labels.append(i)   
        
        for y in range(all_labels.shape[0]):
            for x in range(all_labels.shape[1]):
                if all_labels[y,x] in list_small_labels:
                    all_labels[y,x] = 0
        return all_labels

    def get_nf_label(self, nf_mask):
        label = measure.label(nf_mask)
        new_label = self.remove_small_rois(label)
        nf_mask_filtered = np.bool_(new_label)
        return measure.label(nf_mask_filtered)

    def get_rings(self, nf_label, mask): 

        #dilate mask by one pixel
        mask_dilated = ndimage.binary_dilation(mask)*1

        rings = []
        for i in range(nf_label.max()+1):
            if i > 0:
                #get single neuron        
                nf_object = nf_label==i
                cell_pre = nf_object*1
                #dilate neuron by one pixel
                cell = ndimage.binary_dilation(cell_pre)

                # cut neuron and ring image by 5 pixels around the neuron
                cellbounds = np.where(nf_label==i)
                x_min = cellbounds[0].min()-5
                x_max = cellbounds[0].max()+5
                y_min = cellbounds[1].min()-5
                y_max = cellbounds[1].max()+5
                single_cell = cell[x_min:x_max,y_min:y_max]

                ring = mask_dilated[x_min:x_max,y_min:y_max]

                # find edges of neuron
                edges = segmentation.find_boundaries(single_cell)
                nf_border = edges*single_cell

                # count pixels of edges of neuron
                all_edges = np.ndarray.flatten(nf_border)
                colors, counts = np.unique(all_edges, return_counts = True, axis = 0)
                nf_border_count = counts[1]

                # get overlap of neuron and ring and count the pixels
                overlap = ring+nf_border==2
                all_overlap = np.ndarray.flatten(overlap)
                colors, counts = np.unique(all_overlap, return_counts = True, axis = 0)

                # calculate ring size (overlap/edges)
                if len(counts) ==1:
                    ring_size = 0
                else: 
                    overlap_count = counts[1]
                    ring_size = overlap_count/nf_border_count
                rings.append(ring_size) 
        return rings

