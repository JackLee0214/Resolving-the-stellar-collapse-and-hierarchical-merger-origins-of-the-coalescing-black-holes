
## `Resolving the stellar-collapse and hierarchical-merger origins of the coalescing black holes`


These files include the codes and data to re-produce the results of the work  _Resolving the stellar-collapse and hierarchical-merger origins of the coalescing black holes_, arXiv: [2303.02973](https://arxiv.org/abs/2303.02973)
, [Yin-Jie Li](https://inspirehep.net/authors/1838354) ,  [Yuan-Zhu Wang](https://inspirehep.net/authors/1664025),  [Shao-Peng Tang](https://inspirehep.net/authors/1838355) , and [Yi-Zhong Fan](https://inspirehep.net/authors/1040745)

#### Main requirements
- [BILBY](https://git.ligo.org/lscsoft/bilby)
- [PyMultiNest](https://johannesbuchner.github.io/PyMultiNest/install.html)
- [precession](https://dgerosa.github.io/precession)

#### Data
The events posterior samples are adopted from the [Gravitational Wave Open Science Center](https://www.gw-openscience.org/eventapi/html/GWTC/), here `C01:Mixed` samples are used for analysis and stored in `data/GWTC3_BBH_Mixed_5000.pickle`. 

The injection campaigns `data/o1+o2+o3_bbhpop_real+semianalytic-LIGO-T2100377-v2.hdf5`
Note, one should first download the injection campaign
`o1+o2+o3_bbhpop_real+semianalytic-LIGO-T2100377-v2.hdf5` from [Abbot et al.](https://doi.org/10.5281/zenodo.5546676), 
and set it to `data/`
  
#### Hierarchical Bayesian inference
- Inference with our main model: run the python script `inference.py` , and specify the single-component model or double-component model by setting `label='Single_spin'` or `label='Double_spin'` in the script.

- Inference with the comparing models: run the python script `compared_inference.py`, and specify the population model *PS&LinearCorrelation*, *PS&DoubleSpin*, *PS&DefaultSpin*, or *PP&DefaultSpin* by setting `label='PS_linear'`, `label='PS_bimodal'`,`label='PS_default'`, or `label='PP_default'` in the script.

The inferred results `*.json` will be saved to `results`

#### Results
- `Double_spin_post.pickle` is the posterior samples inferred by the double-component model.
- `Double_spin_informed.pickle` is the events' samples reweighed by the double-component model.

#### Generate figures
Run the python script `figure_script.py`

The figures will be saved to `figures`
  
#### Acknowledgements
The  publicly available code [GWPopulation](https://github.com/ColmTalbot/gwpopulation) is referenced to calculate the variance of log-likelihood in the Monte Carlo integrals, and the [FigureScript](https://dcc.ligo.org/public/0171/P2000434/003/Produce-Figures.ipynb) from [LIGO Document P2000434](https://dcc.ligo.org/LIGO-P2000434/public) is referenced to produced figures in this work.


  


