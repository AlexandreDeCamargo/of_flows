from setuptools import setup, find_packages


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
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research"
        "Programming Language :: Python :: 3.11",
    ],
)