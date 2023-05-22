You're more than welcome to fork this repository and send through pull requests, or file bug reports at the [issues](https://github.com/matt-lourens/hierarqcal/issues) page. For development, we follow the [gitflow](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow#:~:text=What%20is%20Gitflow%3F,lived%20branches%20and%20larger%20commits.) branching strategy and code is formatted with [black](https://black.readthedocs.io/en/stable/). Docstrings follow [the google python style](https://google.github.io/styleguide/pyguide.html).

### Setting up a development environment
To set up a development environment, fork the repository, clone your fork and then install the package in editable mode in a virtual environment:
```bash
cd hierarqcal/
python -m venv .venv
pip install pip --upgrade
pip install -e .[qiskit,cirq,pennylane]
pip install -r requirements_dev.txt
```
There are 4 files under tests/ that is useful for debugging the different frameworks, when debugging your code it's useful to set something similar up. You can create a filename_deprecated.py file which won't be tracked because \*_deprecated\* is in gitignore. There isn't proper unit tests yet but between those test files, the quickstart.ipynb andcore_tutorial.ipynb  notebook there is enough codesamples to test whether things still work as expected and to get a feel for the package. What works for me is: at the top of the file add a typical line of code with a breakpoint and then debug through it. Note that plots only show when you've pasued on a breakpoint and then call the plot (I usually do this through the debug console). On VS code this is done by creating the breakpoint and pressing f5, you can then use f11 to go into the function and f10 to step through it.

If you're working with notebooks, make sure to add the virtualenv to the kernel:
```bash
python -m ipykernel install --user --name=env_generic
```

If you want to build the docs locally, make sure to install sphinx dependencies:
```bash
pip install -r ./docs/requirements.txt
```
And use the following to generate the docs, run:
```bash
cd docs/
sphinx-apidoc -f -E -e -M -P -o source/generated/ ../../hierarqcal/
sphinx-build -b html source build
```
