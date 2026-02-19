# miniscope_fullstack

point the repository to your miniscope files; run the code; get detailed plots and analyses of the mechanical and calcium fluorescence patterns

How to run:
python -m pip install -r requirements.txt
python main.py

To run the streamlit (for an easier UI)
* pip install streamlit
* pip install cellpose
* streamlit run app.py

And enjoy selecting the files you want to run it on from a locally-hosted webapp! No need to go rifling around files.

python main.py will do the following:
1) run trace_extraction, which will give you the peak plots + ROIs annotated on the first frame for verification
1) Provide metrics (e.g. frequency of beating , irregularity, decay90, etc.) from the data 
1) Creative comparative plots for different drug concentrations and types with the organoids.

**Output includes:**
1) Mechanical contractility
2) Calcium transience patterns
3) Metrics, including CD90!
4) In multi-mode, comparative plots of your different testing conditions (different drugs, electrical pacing conditions, etc.)

![DOF with ROI](plots/17_11_28_Dark_Paced_Stablized_NODRUG/upload_fluorescence_spikes.png)
(This is our negative control: shows what a regular commercially-paced organoid will provide as its fluorescent transient peaks.)
From input files that look like this:
![Graphing](plots/17_11_28_Dark_Paced_Stablized_NODRUG/upload_rois_on_first_frame.png)
* Will update with comparative fluorescence patterns for three different drugs and normal soon
