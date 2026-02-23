import setuptools
import os

requirementPath = 'requirements.txt'
reqs = []
if os.path.isfile(requirementPath):
    with open(requirementPath) as f:
        reqs = f.read().splitlines()

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="oulu-human-thermal-sensing",
    version="0.1.0",
    author="Constantino Álvarez Casado, Miguel Bordallo López",
    author_email="constantino.alvarezcasado@oulu.fi",
    description="Contactless cardiorespiratory and sudomotor signal extraction from thermal facial video.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MMLSLab/oulu-human-thermal-sensing",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
    install_requires=reqs,
)
