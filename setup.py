import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="quasar",
    version="0.0.0",
    author="QC Ware Corp.",
    author_email="",
    description="Quasar: an ultralight python-2.7/python-3.X quantum circuit simulator package.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/qcware/quasar",
    packages=setuptools.find_packages(),
    package_data={'': ['*.dat']},
    install_requires=[
        "numpy==1.17.4",
        "scipy==1.3.3",
        "sortedcontainers==2.1.0"
    ],
    extras_require={
        "qiskit": ["qiskit==0.9.0", "networkx==2.3"],
        "cirq": ["cirq==0.5.0"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
