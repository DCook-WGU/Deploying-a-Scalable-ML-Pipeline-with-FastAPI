# Deploying a Scalable ML Pipeline with FastAPI


> NOTE: Working in a command line environment is recommended for ease of use with git and dvc. If on Windows, WSL1 or 2 is recommended.



## Table of contents

- [Preliminary steps](#preliminary-steps)
  * [Fork the Starter Kit](#fork-the-starter-kit)
  * [Create environment](#create-environment)
  * [Quick Start (First-Time Setup)](#quick-start-first-time-setup)
  * [Rebuild vs Start — Which Script to Use?](#rebuild-vs-start--which-script-to-use)
  * [Useful Commands Summary](#useful-commands-summary)
  * [Public Links](#public-links)
  * [Instructions](#instructions)
---

## Preliminary steps

### Supported Operating Systems

This project is compatible with the following operating systems:

- **Ubuntu 22.04** (Jammy Jellyfish) - both Ubuntu installation and WSL (Windows Subsystem for Linux)
- **Ubuntu 24.04** - both Ubuntu installation and WSL (Windows Subsystem for Linux)
- **macOS** - compatible with recent macOS versions

Please ensure you are using one of the supported OS versions to avoid compatibility issues.

### Python Requirement

This project requires **Python 3.10**. Please ensure that you have Python 3.10 installed and set as the default version in your environment to avoid any runtime issues.

---

### Environment Set up (pip or conda)
Two methods have been supplied on for the environment setup, Conda and PIP. For each method, an environment requirements/dependencies file has been provided to assist with the setup. Once you have selected the method, you wish to use, review the information below to determine which file to use for setup. 

* Conda: use the supplied file `environment.yml` to create a new environment with conda
* PIP: use the supplied file `requirements.txt` to create a new environment with pip

For this project's setup, we will be using Conda in junction with Docker. 


### Using WSL and Docker (Recommended Setup)

For consistent development across systems, this project supports running inside a **Docker container on WSL2**.

#### Prerequisites (Windows)
1. Enable WSL2 and install Ubuntu 22.04:
   ```bash
   wsl --install
   ```
2. Install [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop).
   - Enable “Use the WSL 2 based engine”
   - Enable “Integrate with my default WSL distro (Ubuntu)”
3. Verify Docker works inside WSL:
   ```bash
   docker run hello-world
   ```

#### macOS and Linux
Docker runs natively. Ensure Docker Engine is installed and running before proceeding.

---

### Fork the Starter kit
Go to [https://github.com/udacity/Deploying-a-Scalable-ML-Pipeline-with-FastAPI](https://github.com/udacity/Deploying-a-Scalable-ML-Pipeline-with-FastAPI)
and click on `Fork` in the upper right corner. This will create a fork in your Github account, i.e., a copy of the
repository that is under your control. Now clone the repository locally so you can start working on it:

```bash
git clone https://github.com/[your github username]/Deploying-a-Scalable-ML-Pipeline-with-FastAPI.git

```

and go into the repository:

```bash
cd Deploying-a-Scalable-ML-Pipeline-with-FastAPI
```

Commit and push to the repository often while you make progress towards the solution. Remember 
to add meaningful commit messages.

---

### Create Environment (Using Docker)

Instead of manually creating a Conda environment, this project provides a script to build and launch a Docker container with the full setup.

From your Ubuntu (WSL) or macOS/Linux terminal:

```bash
# Navigate to the folder where you cloned the repository
cd Deploying-a-Scalable-ML-Pipeline-with-FastAPI

# Build the Docker image and start the container
./rebuild.sh
```

This script will:
1. Stop and remove any old container (if it exists)
2. Build a new Docker image using the included Dockerfile
3. Start a container in the background
4. Print a command you can use to connect to the container

Once complete, connect to your running container:
```bash
docker exec -it mlops-project-two bash
```

Then activate the Conda environment inside the container:
```bash
conda activate mlops-dev-project-two
```

Your environment is now ready for MLflow and W&B usage.

---

### Quick Start (First-Time Setup)

If this is your first time setting up the project, follow these steps:

```bash
# 1. Clone this repository
git clone https://github.com/DCook-WGU/Deploying-a-Scalable-ML-Pipeline-with-FastAPI.git

cd Deploying-a-Scalable-ML-Pipeline-with-FastAPI

# 2. Build and start the container
./rebuild.sh

# 3. Enter the running container
docker exec -it mlops-project-two bash

# 4. Activate the environment (It should auto activate this env for you, but just in case.)
conda activate mlops-dev-project-two

# 5. Run the pipeline
mlflow run .
```

---

### 🧠 Rebuild vs Start — Which Script to Use?

| Script | Builds Image? | Removes Old Container? | Runs Interactively? | Use When |
|--------|----------------|------------------------|---------------------|----------|
| **`./rebuild.sh`** | ✅ Yes | ✅ Yes | 💤 Detached (background) | First-time setup or after Dockerfile/environment changes |
| **`./start.sh`** | ❌ No | ❌ No | ✅ Interactive (`-it`) | Reusing an existing image for daily development |

### Common Workflows

**First-Time Setup**
```bash
./rebuild.sh
docker exec -it mlops-project-two bash
conda activate mlops-dev-project-two
```

> Note: It should auto activate the conda env, but if it doesn't then use the command above.

**Daily Use (No rebuild needed)**
```bash
./start.sh
```

---

### 📦 Useful Commands Summary

| Task | Command |
|------|----------|
| Build & start container | `./rebuild.sh` |
| Enter running container | `docker exec -it mlops-project-two bash` |
| Restart without rebuild | `./start.sh` |
| Run pipeline | `mlflow run .` |
| Check Conda env | `conda env list` |

---

### Public Links

- **GitHub Repository:** [https://github.com/DCook-WGU/Deploying-a-Scalable-ML-Pipeline-with-FastAPI](https://github.com/DCook-WGU/Deploying-a-Scalable-ML-Pipeline-with-FastAPI)  
- **Latest Release:** [https://github.com/DCook-WGU/Deploying-a-Scalable-ML-Pipeline-with-FastAPI/releases/latest](https://github.com/DCook-WGU/Deploying-a-Scalable-ML-Pipeline-with-FastAPI/releases/latest)  


---

### Instructions:

#### Repositories
* Create a directory for the project and initialize git.
    * As you work on the code, continually commit changes. Trained models you want to use in production must be committed to GitHub.
* Connect your local git repo to GitHub.
* Setup GitHub Actions on your repo. You can use one of the pre-made GitHub Actions if at a minimum it runs pytest and flake8 on push and requires both to pass without error.
    * Make sure you set up the GitHub Action to have the same version of Python as you used in development.

#### Data
* Download census.csv and commit it to dvc.
* This data is messy, try to open it in pandas and see what you get.
* To clean it, use your favorite text editor to remove all spaces.

#### Model
* Using the starter code, write a machine learning model that trains on the clean data and saves the model. Complete any function that has been started.
* Write unit tests for at least 3 functions in the model code.
* Write a function that outputs the performance of the model on slices of the data.
    * Suggestion: for simplicity, the function can just output the performance on slices of just the categorical features.
* Write a model card using the provided template.

#### API Creation
*  Create a RESTful API using FastAPI this must implement:
    * GET on the root giving a welcome message.
    * POST that does model inference.
