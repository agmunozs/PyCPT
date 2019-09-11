# PyCPT - seasonal and sub-seasonal versions
Python interface for the CPT command line version, for seasonal and sub-seasonal skill assessment and forecast experiments

PyCPT_s2svX: version X of the sub-seasonal PyCPT notebook

PyCPT_seavX: version X of the seasonal PyCPT notebook

## Authors
Ángel G. Muñoz (agmunoz@iri.columbia.edu), Andrew W. Robertson (awr@iri.columbia.edu), Simon J. Mason

## Contributors
Jeff Turmelle (IRI), Thea Turkington (NEA), Xandre Chourio (IRI), Rémi Cousin (IRI), Nachiketa Acharya (IRI), Asher Siebert (IRI)

## Acknowledgements
James Doss-Gollin (Columbia Water Center) developed the original download Python functions.

Part of the effort to develop PyCPT is funded by the Columbia World Project "Adapting Agriculture to Climate Today, for Tomorrow" (ACToday), and NOAA MAPP's projects NA18OAR4310275 (Muñoz) and NA16OAR4310145 (Robertson).

## Instructions:
1. Download and compile CPT (https://iri.columbia.edu/cpt). Use version > 16.0 for seasonal, and version => 15.7.10 for sub-seasonal. Take note of the PATH to CPT.
2. Clone or download the function and notebook files in the same (working) folder, open the notebook, modify the namelist to include, for example, your PATH to CPT and working directory. Check dates (especially if interested in realtime cases).

3. [ONLY needed for the s2s version] Create the file .IRIDLAUTH in the same folder. Its content must be the Data Library S2S key obtained via help@iri.columbia.edu
4. Run the notebook.
