# miniscope_fullstack

General notes:
* trying to run everything in python so it can sustain itself
  
1) point the repository to your miniscope .avi files; 
2) The .avi files will be converted to .tiff ones.
3) call ImageJ in order to identify ROIs, using F/F0 ratios
4) export fluorescent intensity traces for each ROI as CSVs
5) identify peaks in calcium fluorescence, segment each 'heartbeat', normalize each trace, and average results per ROI
6) calculate decay 50, CD90, etc. to quantify drug effects on calcium dynamics.

How to use:
* clone the github locally
* run environment file
* change folder path in changeable_vars.py in order to match where your miniscope files are
* python main.py
