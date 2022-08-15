FROM conda/miniconda3:latest

RUN     conda install -c conda-forge -y mamba
RUN     mamba install -y numpy scipy mkl
RUN     conda install -c conda-forge -y scikit-learn
RUN     mamba install -c conda-forge -y theano-pymc
RUN     mamba install -c conda-forge -y pymc3
RUN     mamba install -c conda-forge -y python-graphviz
RUN     mamba install -c conda-forge -y notebook
