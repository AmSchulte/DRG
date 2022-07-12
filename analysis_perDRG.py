import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

class GroupData:
    def __init__(self, experiment_group, drg_position, number_of_rats):
        self.experiment_group = experiment_group
        self.drg_position = drg_position

        if 'Sham' in experiment_group[0]['path']:
            self.name = 'Sham '+ drg_position
        elif 'SNI' in experiment_group[0]['path']:
            self.name = 'SNI '+ drg_position

        #get parameters for each image and calculate the mean of all images belonging to one DRG
        self.get_numbers(number_of_rats)


    def get_numbers(self, number_of_rats):
        self.ring_ratios_gfap = []
        self.ring_ratios_gs = []
        self.ring_ratios_ring = []
        self.gfap_intensities = []
        self.gs_intensities = []
        self.gfap_intensities_norm = []
        self.gs_intensities_norm = []
        self.gfap_integrated_density = []
        self.gs_integrated_density = []
        self.gfap_intensity_per_area = []
        self.gs_intensity_per_area = []
        self.small_neurons_percentage = []
        self.bigger_neurons_percentage = []
        self.small_neurons_gs_ring = []
        self.small_neurons_gfap_ring = []
        self.bigger_neurons_gs_ring = []
        self.bigger_neurons_gfap_ring = []
        self.gs_overlaps = []
        self.gfap_overlaps = []
        self.neuronal_area_per_tissue = []
        self.neurons_per_tissue = []
        self.gs_area_per_tissue = []
        self.gfap_area_per_tissue = []
        self.ring_area_per_tissue = []
        self.gs_area_per_neurons = []
        self.gfap_area_per_neurons = []
        self.ring_area_per_neurons = []

        self.ring_ratios_gfap_std = []
        self.n_number = []
        self.neuron_bins = None
        self.neuron_counts = []

        self.all_neurons = []
        self.tissue_pixels = []
        self.neurons_per_neuronal_area = []
        self.neurons_corrected = []

        for n in range(number_of_rats):
            rat = 'Ratte '+str(n+1)
            for i, drg in enumerate(self.experiment_group):
                if self.drg_position in drg['group'] and rat in self.experiment_group[i]['path']:
                    self.get_numbers_for_each_image(drg)

    # actually, it is mean values for each DRG
    def get_numbers_for_each_image(self, drg):
        gfap_overlap = []
        gs_overlap = []
        neuronal_area_tissue = []
        neurons_per_tissue = []
        gs_area_tissue = []
        gfap_area_tissue = []
        ring_area_tissue = []
        gs_area_neurons = []
        gfap_area_neurons = []
        ring_area_neurons = []
        ring_ratio_gfap = []
        ring_ratio_gs = []
        ring_ratio_ring = []
        small_neurons_gs_r = []
        small_neurons_gfap_r = []
        bigger_neurons_gs_r = []
        bigger_neurons_gfap_r = []
        small_neuron_percentage = []
        bigger_neuron_percentage = []
        gfap_intensity = []
        gs_intensity = []
        gfap_intensity_norm = []
        gs_intensity_norm = []
        gfap_int_den = []
        gs_int_den = []
        gfap_int_per_area = []
        gs_int_per_area = []
       

        amount_neurons = []
        tissue_areas = []

        neuron_counts = []

        neurons_per_neuronal_area = []
        neurons_corrected = []

        for gfap, gs, ring, nf, gfap_i, gs_i, gfap_norm_i, gs_norm_i, gfap_ol, gs_ol, nf_area, gs_area, gfap_area, ring_area, gs_neuron, gfap_neuron, ring_neuron, tissue_px, gfap_px, gs_px, gfap_int_area, gs_int_area in zip(drg['gfap_rings'], 
        drg['gs_rings'], drg['ring_rings'], drg['cell_size_neurons'], drg['gfap_intensities'], drg['gs_intensities'], drg['gfap_intensities_norm'], drg['gs_intensities_norm'], drg['gfap_overlaps'], drg['gs_overlaps'],
        drg['neuronal_area_per_tissue'], drg['gs_area_per_tissue'], drg['gfap_area_per_tissue'], drg['ring_area_per_tissue'], drg['gs_area_per_neurons'], drg['gfap_area_per_neurons'], drg['ring_area_per_neurons'], 
        drg['tissue_pixel'], drg['gfap_pixel'], drg['gs_pixel'], drg['gfap_intensities_area'] ,drg['gs_intensities_area']):

            # get the number of rings that are bigger than 0%/50& around the neurons
            number_of_rings_gfap = len([i for i in gfap if i>0])
            number_of_rings_gs = len([i for i in gs if i>0])
            number_of_rings_ring = len([i for i in ring if i>0])
            number_of_neurons = len(nf)

            # get neuronal number per tissue area
            # 1px = 0.000000824464mm^2
            neurons_per_area = number_of_neurons/(tissue_px*0.000000824464)
            neurons_per_tissue.append(neurons_per_area)

            amount_neurons.append(number_of_neurons)
            tissue_areas.append(tissue_px)

            # only include images which contain neurons
            if nf_area > 0 and number_of_neurons > 0:               
                neurons_per_neuronal= number_of_neurons/(nf_area*tissue_px*0.000000824464)
                neurons_per_neuronal_area.append(neurons_per_neuronal)
                neurons_corr = (1/nf_area) * neurons_per_area
                neurons_corrected.append(neurons_corr)
                
                # determine percentage of neural sizes in 100µm2 steps for histogram
                # pixel^2 -> µm^2 (1px = 0.908µm)
                nf_sizes = np.array(nf)*0.814
                # neurons bigger than 3000 µm2 are counted as size 3000
                nf_sizes[nf_sizes>3000] = 3000
                counts, bins = np.histogram(nf_sizes, bins=15, range=(0,3000))
                counts_percent = counts/counts.sum()*100  

                neuron_counts.append(counts_percent.tolist())

            if not np.isnan(gfap_ol):
                gfap_overlap.append(gfap_ol*100)
            if not np.isnan(gs_ol):
                gs_overlap.append(gs_ol*100)
            neuronal_area_tissue.append(nf_area*100)
            gs_area_tissue.append(gs_area*100)
            gfap_area_tissue.append(gfap_area*100)
            ring_area_tissue.append(ring_area*100)
            gs_area_neurons.append(gs_neuron*100)
            gfap_area_neurons.append(gfap_neuron*100)
            ring_area_neurons.append(ring_neuron*100)            

            # get neuron size ratios in percent
            # 850 pixel corresponds to 700 µm^2 area (30µm diameter)
            # only include images with neurons
            if number_of_neurons > 0:
                ratio_gfap = (number_of_rings_gfap/number_of_neurons)*100
                ring_ratio_gfap.append(ratio_gfap)
                ratio_gs = (number_of_rings_gs/number_of_neurons)*100
                ring_ratio_gs.append(ratio_gs)
                ratio_ring = (number_of_rings_ring/number_of_neurons)*100
                ring_ratio_ring.append(ratio_ring)

                small_neurons = []
                small_neurons_gs = []
                small_neurons_gfap = []
                bigger_neurons = []
                bigger_neurons_gs = []
                bigger_neurons_gfap = []
                for neuron, gs_ring, gfap_ring in zip(nf, gs, gfap):
                    if neuron < 850:
                        small_neurons.append(neuron)
                        if gs_ring > 0:
                            small_neurons_gs.append(gs_ring)
                        if gfap_ring > 0:
                            small_neurons_gfap.append(gfap_ring)
                    elif neuron >= 850:
                        bigger_neurons.append(neuron)
                        if gs_ring > 0:
                            bigger_neurons_gs.append(gs_ring)
                        if gfap_ring > 0:
                            bigger_neurons_gfap.append(gfap_ring)
                
                number_of_small_neurons = len(small_neurons)
                if number_of_small_neurons>0:
                    small_neurons_gs_ratio = (len(small_neurons_gs)/number_of_small_neurons)*100
                    small_neurons_gfap_ratio = (len(small_neurons_gfap)/number_of_small_neurons)*100
                    small_neurons_gs_r.append(small_neurons_gs_ratio)
                    small_neurons_gfap_r.append(small_neurons_gfap_ratio)
                
                number_of_bigger_neurons = len(bigger_neurons)
                if number_of_bigger_neurons>0:
                    bigger_neurons_gs_ratio = (len(bigger_neurons_gs)/number_of_bigger_neurons)*100
                    bigger_neurons_gfap_ratio = (len(bigger_neurons_gfap)/number_of_bigger_neurons)*100
                    bigger_neurons_gs_r.append(bigger_neurons_gs_ratio)
                    bigger_neurons_gfap_r.append(bigger_neurons_gfap_ratio)

                small_neurons_ratio = number_of_small_neurons/len(nf)*100
                bigger_neurons_ratio = len(bigger_neurons)/len(nf)*100
                small_neuron_percentage.append(small_neurons_ratio)
                bigger_neuron_percentage.append(bigger_neurons_ratio)

            
            gfap_intensity.append(gfap_i)
            gfap_intensity_norm.append(gfap_norm_i)
            gs_intensity.append(gs_i)
            gs_intensity_norm.append(gs_norm_i)
            
            #integrated density
            if gfap_int_area>0:
                gfap_int_den.append(gfap_int_area*gfap_px)
                gfap_int_per_area.append(gfap_int_area)
            else:
                gfap_int_den.append(0)
                gfap_int_per_area.append(0)
            if gs_int_area>0:
                gs_int_den.append(gs_int_area*gs_px)
                gs_int_per_area.append(gs_int_area)
            else:
                gs_int_den.append(0)
                gs_int_per_area.append(0)
        
        self.ring_ratios_gfap.append(np.mean(ring_ratio_gfap))
        self.ring_ratios_gs.append(np.mean(ring_ratio_gs))
        self.ring_ratios_ring.append(np.mean(ring_ratio_ring))
        self.gfap_intensities.append(np.mean(gfap_intensity))
        self.gs_intensities.append(np.mean(gs_intensity))
        self.gfap_intensities_norm.append(np.mean(gfap_intensity_norm))
        self.gs_intensities_norm.append(np.mean(gs_intensity_norm))
        self.small_neurons_percentage.append(np.mean(small_neuron_percentage))
        self.bigger_neurons_percentage.append(np.mean(bigger_neuron_percentage))
        self.small_neurons_gs_ring.append(np.mean(small_neurons_gs_r))
        self.small_neurons_gfap_ring.append(np.mean(small_neurons_gfap_r))
        self.bigger_neurons_gs_ring.append(np.mean(bigger_neurons_gs_r))
        self.bigger_neurons_gfap_ring.append(np.mean(bigger_neurons_gfap_r))
        self.gs_overlaps.append(np.mean(gs_overlap))
        self.gfap_overlaps.append(np.mean(gfap_overlap))
        self.neuronal_area_per_tissue.append(np.mean(neuronal_area_tissue))
        self.neurons_per_tissue.append(np.mean(neurons_per_tissue))
        self.gs_area_per_tissue.append(np.mean(gs_area_tissue))
        self.gfap_area_per_tissue.append(np.mean(gfap_area_tissue))
        self.ring_area_per_tissue.append(np.mean(ring_area_tissue))
        self.gs_area_per_neurons.append(np.mean(gs_area_neurons))
        self.gfap_area_per_neurons.append(np.mean(gfap_area_neurons))
        self.ring_area_per_neurons.append(np.mean(ring_area_neurons))

        self.ring_ratios_gfap_std.append(np.std(np.array(ring_ratio_gfap)))
        self.n_number.append(len(ring_ratio_gfap))

        self.all_neurons.append(np.mean(amount_neurons))
        self.tissue_pixels.append(np.mean(tissue_areas))
        self.gfap_integrated_density.append(np.mean(gfap_int_den))
        self.gs_integrated_density.append(np.mean(gs_int_den))
        self.gfap_intensity_per_area.append(np.mean(gfap_int_per_area))
        self.gs_intensity_per_area.append(np.mean(gs_int_per_area))

        self.neurons_per_neuronal_area.append(np.mean(neurons_per_neuronal_area))
        self.neurons_corrected.append(np.mean(neurons_corrected))

        self.neuron_counts.append(np.mean(neuron_counts, axis=0).tolist())
        self.neuron_bins = bins

