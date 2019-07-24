== setup sphinx ==
1. Install sphinx through this site: 
http://www.sphinx-doc.org/en/master/usage/installation.html
2. Run "sphinx-quickstart" in comandline and answer all questions prompted.
3. Install the theme by "pip install sphinx-rtd-theme".

== make ==
3. Sphinx takes .rst files and generates .html files. 
All the .rst are in "source" folder. All .html are in "build" folder.
4. Run "make html" in commandline in the docs folder, it will generate html files in the "build" folder. 
5. If contents in autosummary fields of a .rst file (say circuit.rst) have been changed, run "sphinx-autogen circuit.rst". 
It will generate a .rst file for each item in autosummary fields.

== add contents ==
6. To append the newly created .rst file, type the file name of the .rst file in the toc tree in the index.rst.

== github ==
Usually, people only put source folder in repo, but not the build folder.