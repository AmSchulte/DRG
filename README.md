## Bioimage analysis of cellular changes in rat DRG after SNI
This repository contains the Python implementation for analyzing DRG tile scan images, as described in the paper:

PAIN paper

### Outline of analysis steps:
<a href="url"><img src="https://github.com/AmSchulte/DRG/blob/main/analysis_graph.png" width="600" ></a>


Segmentations of images (tif) were previously prediceted with deepflash2 and saved as png files.
All data, including image annotations used for training of deep learning models, is available at https://doi.org/10.5281/zenodo.6546069.
Locally, images and corresponding masks can be viewed with Streamlit (streamlit.py).
Online, example segementation can be viewed at https://share.streamlit.io/amschulte/drg/main.
The app (streamlit_app.py) is lauched over Streamlit, visualized data is uploaded on github.

Analysis steps:
1. Calculation of parameters per image set:
   - class with calculation script and functions: DrgData in drg.py
   - execution in Jupyter notebook (Pipeline_DRG_results.ipynb)
   - for each image set (images and masks of NF, GS, and GFAP), a dictionary containing the analysis group (L4CL/L4IL/L5CL/L5IL), path, and parameters is created
2. Save to JSON file
   - in Pipeline_DRG_results.ipynb
   - for each experiment group, a list of dictionaries with the analysis results is saved in a json file (D7_Sham_area.json, D7_SNI_area.json, D14_Sham_area.json, D14_SNI_area.results.json)
   - rationale: computation of results for each experiment group took about 4h, making it necessarry to save intermediate results before final evaluation and visualization 
3. Calculation of parameters per DRG:
   - with GroupData class in analysis_perDRG.py
   - calculation of final parameters, averated for each DRG
   - used before visulation and calculation of statistics
4. Visualization:
   - Boxplots: Boxplots_d7+d14.ipynb, Boxplots_d7.ipynb, Boxplots_d14.ipynb, Boxplots_L4+L5_d7.ipynb, Boxplots_L4+L5_d14.ipynb 
   - Histogramms: Histogramm_neurons_d7.ipynb, Histogramm_neurons_d14.ipynb 
5. Statistics:
   - in Statistic.ipynb
   - saved as an excel file



### Structure of bioimage analysis of rat DRG:
<a href="url"><img src="https://github.com/AmSchulte/DRG/blob/main/analysis_details_graph.png" width="800" ></a>
