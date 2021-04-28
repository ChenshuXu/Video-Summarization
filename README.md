# Video-Summarization
# Set up the Virtual Environment

Download and install the latest Python3 supported version of Anaconda for your OS [here](https://www.anaconda.com/download).

From the local course repository directory you created when you cloned the remote, create the virtual environment by running
```
conda env create -f environment.yml
```

**Resource:** Anaconda Documentation - [Creating an environment from an environment yml file](https://conda.io/docs/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file)

You can then activate the virtual environment from a terminal by executing the following command:

```
conda activate Video-Summarization
```

You can deactivate the virtual environment by executing:

```
conda deactivate
```

You can remove the virtual environment by executing:
```
conda remove --name Video-Summarization --all
```

Once the virtual environment has been activated, you can execute code from the same terminal. Or use in other Python IDEs, such as PyCharm, VSCode.

Then create a folder `input`, put all `audio` folder and `frames` folder into this folder.

If added any new libraries, create new environment yml file by executing:
```
conda env export > environment.yml
```