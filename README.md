# WGNN-DTA
a method for CPI and DTA prediction.
## 1.Grpah construction
The first thing to do for WGNN_DTA is to construct the protein weighted graph, we prepare a script to do this, just run it:
 python graph_prepare.py '''
And the script can generate graphs for 4 datasets, which are human, celegants, davis and kiba, just simply edit the dataset name in the script.
## 2.Run CPI training
' python train_cpi.py 0 0 0 '
where there are 3 parameters:
first parameter: datasets, and 0 for human, 1 for celegants;
second parameter: gpu number, the selected gpu, change the script if you have more gpus than two;
third parameter: ratio, the ratio of negative and positive, 0 for 1:1, 1 for 3:1 and 2 for 5:1.
## 3.Run DTA training
''' python train_dta.py 0 0 '''
where there are 2 parameters:
first parameter: datasets, and 0 for davis, 1 for kiba;
second parameter: gpu number, the selected gpu, change the script if you have more gpus than two.

The codes are based on GNN, which are easy to entend to your demands (even struture-based methods). You can change the codes according to your needs.
