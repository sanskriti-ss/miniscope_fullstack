# miniscope_fullstack

point the repository to your miniscope files; run the code; get detailed plots and analyses of the mechanical and calcium fluorescence patterns

How to run:
python -m pip install -r requirements.txt
python main.py


python main.py will do the following:
1) run trace_extraction, which will give you the peak plots + ROIs annotated on the first frame for verification
2) give you plots (decay 90, etc) from the data (still in progress; insufficient data collection as of last update).


Input files will look like this:
![DOF with ROI](plots/17_11_28_Dark_Paced_Stablized_NODRUG/upload_fluorescence_spikes.png)
Output files may look like this:
![Graphing](plots/17_11_28_Dark_Paced_Stablized_NODRUG/upload_rois_on_first_frame.png)
* Will update with comparative fluorescence patterns for three different drugs and normal soon


Main to-dos:
* add mechanical contractility modules after validating the data files.



might want to change these depending on the size of the expected blobs:
MIN_AREA = 600
MAX_AREA = 50_000
