# PyCPT - v1.2
Python interface for the CPT command line version, for seasonal and sub-seasonal skill assessment and forecast experiments

PyCPT_s2svX: version X of the sub-seasonal PyCPT notebook

PyCPT_seavX: version X of the seasonal PyCPT notebook

## Authors
Ángel G. Muñoz (agmunoz@iri.columbia.edu) and Andrew W. Robertson (awr@iri.columbia.edu)

## Acknowledgements
Simon J. Mason (IRI) for CPT core development.

Rémi Cousin (IRI) for key Ingrid code development and support.

Xandre Chourio (IRI) for Windows and code support, and testing.
James Doss-Gollin (Columbia Water Center) developed the original download Python functions.
Thea Turkington (NEA) for several and intense tesrting. 
Part of the effort to develop PyCPT is funded by the Columbia World Project "Adapting Agriculture to Climate Today, for Tomorrow" (ACToday), and NOAA MAPP's projects NA18OAR4310275 (Muñoz) and NA16OAR4310145 (Robertson).

## Instructions:
1. Download and compile CPT (https://iri.columbia.edu/cpt). Take note of the PATH to CPT.
2. Clone or download the function and notebook files in the same (working) folder, open the notebook, modify the namelist to include, for example, your PATH to CPT and working directory. Check dates (especially if interested in realtime cases).
3. Create the file .IRIDLAUTH in the same folder. Its content must be the Data Library S2S key obtained via help@iri.columbia.edu
4. Run the notebook.
