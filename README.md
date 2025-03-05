# SISAP 2025 Challenge: Working example on Julia 

This repository is a working example for the SISAP 2025 Indexing Challenge <https://sisap-challenges.github.io/>, working with Julia and GitHub Actions, as specified in Task's descriptions. It is based on the previous year example.

## Steps for running
It requires a working installation of Julia (better works with `v1.10.8`. _Note that `v1.11` series have performance regressions for our similarity search package_. 
You can download Julia from <https://julialang.org/downloads/>. You also need `git` tools and internet access for cloning, install package dependencies and downloading datasets.

The steps are the following:

1. Clone the example repository
2. Fetch the datasets
3. Instantiate the project
4. Run
5. Evaluation

the fullset of instructions are listed in the GitHub Actions workflow

<https://github.com/sisap-challenges/sisap25-example-julia/blob/main/.github/workflows/ci.yml>

Note that you will need to adjust your scripts to hold the correct hyperparameters for any benchmark you use, in particular, `pubmed23` and `gooaq` which will be used in the testing stage.

### Clone this repository
```base
git clone https://github.com/sisap-challenges/sisap25-example-julia
cd sisap25-example-julia
```

### Fetch the datasets
You can clone our dataset repository or fetch dataset by dataset. For instance, if you want to test this example you can download the smallest dataset CCNEWS dataset, even with a reduced precision (fp16) and with a shortened gold-standard (the main files contain $k=1000$ nearest neighbors).

Run the `prepare-data.sh` script to download the necessary CCNEWS data. Note that we expect to use this small dataset for your GitHub Actions.

For your participation and experimentation you may want to clone the entire dataset repository `https://huggingface.co/datasets/sadit/SISAP2025/`; 
as follows:

```bash
git clone https://huggingface.co/datasets/sadit/SISAP2025
```

The current example look for a `data` directory so you can rename the local repository or do a symbolic link
```bash
ln -s SISAP2025 data 
```

### Instantiate the project
Julia requires to prepare the working directory through an `instantiation`, _a.k.a. installing dependencies_, as follows:

```bash
JULIA_PROJECT=. JULIA_NUM_THREADS=8 julia -e 'using Pkg; Pkg.instantiate()'
```

You need internet access for this step.

### Run
A similar procedure is needed to run; note that Julia may compile many packages in the first run, so please be patient.
```bash
JULIA_PROJECT=. JULIA_NUM_THREADS=8 julia -L sisap2025.jl -e 'main_task1(); main_task2()'
```

You should modify the number of threads to adapt your hardware; you can also try to add optimization flags to julia, e.g., call it `julia -O3 -Cnative`. 

### Evaluation
```bash
JULIA_PROJECT=. julia -L eval.jl -e 'eval_task1()'
JULIA_PROJECT=. julia -L eval.jl -e 'eval_task2()'
```

Two result files will be created: `result-task1.csv` and `result-task2.csv`.

## How to take this to create my own system
You can fork this repository and polish it to create your solution. Please also take care of the ci workflow (see below).

## GitHub Actions: Continuous integration 

You can monitor your runnings in the "Actions" tab of the GitHub panel: for instance, you can see some runs of this repository:
<https://github.com/sisap-challenges/sisap25-example-julia/actions>

 
