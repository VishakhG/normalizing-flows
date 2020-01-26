# Normalizing flows

Attempting to implement the potential function experiments from:

```
Danilo Jimenez Rezende and Shakir Mohamed. Variational inference with normalizing
flows. In Proceedings of the 32nd International Conference on Machine Learning, pages
1530–1538, 2015.
```
Other reference:

```
Papamakarios, George, et al. Normalizing Flows for Probabilistic Modeling and Inference. Dec. 2019. arxiv.org, https://arxiv.org/abs/1912.02762v1.

```

To reproduce plots run `exp/run_2d_potential_exp.sh` or take a look at `src/fit_flows.py`.

Target densities, corresponding to the 4 potentials from the paper:

![target densities](https://github.com/VishakhG/normalizing-flows/blob/master/assets/all_potentials.png)


Samples from a 2-D diagonal gaussian passed through 32 learned Planar flows:

![potential 1](https://github.com/VishakhG/normalizing-flows/blob/master/assets/pot_1_32.png)
![potential 2](https://github.com/VishakhG/normalizing-flows/blob/master/assets/pot_2_32.png)
![potential 3](https://github.com/VishakhG/normalizing-flows/blob/master/assets/pot_3_32.png)
![potential 4](https://github.com/VishakhG/normalizing-flows/blob/master/assets/pot_4_32.png)

