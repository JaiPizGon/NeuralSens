#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['']

test_requirements = ['pytest>=3', ]

setup(
    author="Jaime Pizarroso Gonzalo",
    author_email='jpizarrosogonzalo@gmail.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    description="Analysis functions to quantify inputs importance in neural network models.",
    entry_points={
        'console_scripts': [
            'neuralsens=neuralsens.cli:main',
        ],
    },
    install_requires=requirements,
    license="GNU General Public License v3",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    name='neuralsens',
    package_dir={"": "src"},
    packages=find_packages(where="src",include=['neuralsens', 'neuralsens.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/JaiPizGon/neuralsens',
    version='0.0.4.dev10',
    zip_safe=False,
    keywords="neural networks, mlp, sensitivity, XAI, IML, neuralsens",
)
