# The COVID-19 research outbreak: how the pandemic culminated in a surge of new researchers &nbsp; &nbsp; [![Zenodo](https://zenodo.org/badge/DOI/xxx.svg)](https://doi.org/xxxx)



Official code repository for the paper "The COVID-19 research outbreak: how the pandemic culminated in a surge of new researchers"

## Data preprocessing
1. Create the `newcomers` conda environment from `environment.yml` by running 
```
conda env create -f environment.yml
```

2. Create data directory by executing the following command:
```
mkdir -p data
```

3. Download the OpenAlex slices from [Zenodo](https://doi.org/xxxx) inside the `data` directory. 


4. Extract `COVID.zip` zipped slices, so you should have the following files inside `data`: 
`works.parquet`, `works_authors.parquet`, and `works_concepts.parquet`.   

## Running the experiments and generate plots 
* Run `main-experiment.ipynb` 
* More info coming soon..

## More statistical analysis
* Run `stat-analysis.ipynb`
* More info coming soon..
