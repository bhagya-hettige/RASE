# RASE
Robust Attribute and Structure Preserving Graph Embedding

Split the dataset
-----------------
python split_data.py --name=<name> --p_val=<p_val> --p_test=<p_test>

Train with RASE
---------------------
python train.py <name> <lace|glace> --proximity=<first-order|second-order> --is_all=<all_edges> etc.

