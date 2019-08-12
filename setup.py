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
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
