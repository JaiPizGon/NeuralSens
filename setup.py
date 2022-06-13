import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="neuralsens",
    version="0.0.4.dev5",
    # 1.2.0.dev1  # Development release
    # 1.2.0a1     # Alpha Release
    # 1.2.0b1     # Beta Release
    # 1.2.0rc1    # Release Candidate
    # 1.2.0       # Final Release
    # 1.2.0.post1 # Post Release
    # 15.10       # Date based release
    # 23          # Serial release
    author="Jaime Pizarroso Gonzalo",
    author_email="jpizarrosogonzalo@gmail.com",
    description="Analysis functions to quantify inputs importance in neural network models.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JaiPizGon/NeuralSens",
    project_urls={
        "Bug Tracker": "https://github.com/JaiPizGon/NeuralSens/issues",
        "Acknowledgements": "https://www.iit.comillas.edu/",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
    keywords="neural networks, mlp, sensitivity, XAI, IML",
    extras_require={"calculus": ["numpy", "pandas",]},
)
