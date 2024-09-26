## Neighborhood-enhanced Graph Self-Supervised Learning for Recommendation
<font color='red'>The implementation code will be released after the acceptance of the paper.</font>


## Running environment
We develop our codes in the following environment:

- torch == 1.13.1
- faiss==1.7.3
- numba==0.58.1
- numpy==1.24.3
- scipy==1.10.1

## You can run other datasets by modifying some parameters in the NeSR.conf file

amazon-book

NeSR=-n_layer 2 -str_reg 0.1 -sim_reg 0.01 -tau 0.1 -walk 5 -num_clusters 500

Last-FM

NeSR=-n_layer 2 -str_reg 0.5 -sim_reg 0.01 -tau 0.1 -walk 5 -num_clusters 500

amazon-kindle

NeSR=-n_layer 1 -str_reg 0.01 -sim_reg 0.001 -tau 0.1 -walk 5 -num_clusters 500

## How to run the codes


```python
python main.py
