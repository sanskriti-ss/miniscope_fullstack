# miniscope_fullstack

point the repository to your miniscope files; run the code; get detailed plots and analyses of the mechanical and calcium fluorescence patterns

How to run:
python -m pip install -r requirements.txt
python main.py


python main.py will do the following:
1) run trace_extraction, which will give you the peak plots + ROIs annotated on the first frame for verification
2) give you plots (decay 90, etc) from the data



might want to change these depending on the size of the expected blobs:
MIN_AREA = 600
MAX_AREA = 50_000
