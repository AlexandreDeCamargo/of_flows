from setuptools import setup, find_packages

# Parse requirements from file
with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

short_description = \
    """
    A library for learning the ground state energy using orbital free density functional theory
    with continuous normalizing flows.
    """

setup(
    name='of_flows',
    version='0.1',
    author='Alexandre de Camargo',
    author_email='decamara@mcmaster.ca',
    description=short_description,
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/AlexandreDeCamargo/of_flows',
    packages=find_packages(),
    install_requires=requirements,
    extras_require={
        "examples": [
            "matplotlib>=3.10.0", "pandas>=2.0.3", "chex>=0.1.89"
        ],
    },  
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research"
        "License :: OSI Approved ::  MIT License",
        "Programming Language :: Python :: 3.11",
    ],
)