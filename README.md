# Chemlab
This is a private reposity to save my scripts or other code stuff.


### /script/scan_2d
A script written in 2025.4. The scripts use qchem (can support other input, but 
I didn't write class for other software right now). It can generate 2d scan files with a reference
file and scan variables, use bash script to run these jobs in order, and analysis the 2d scan result.

the 2d scan work from optimize the first row in parallel, and then extract the former structure to run the next job in
same row. You can exchange the col and row in bash to reverse it.

To use the script, copy the scan_2d folder and put the reference file that contains moleculer
structure and standard input details. for qchem, this should include at lease $rem$ and $opt2$

The detailed tutorial is the notebook in the scan_2d folder.

### util

Functions to handle qchem outputs

