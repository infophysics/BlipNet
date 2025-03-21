from setuptools import find_packages
from setuptools import setup

with open("README.md", "r") as file:
    long_description = file.read()

setup(
    # name
    name='blipnet',

    # current version
    #   MAJOR VERSION:  00
    #   MINOR VERSION:  01
    #   Maintenance:    00
    version='00.01.00',

    # descriptions
    description='BlipNet',
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords='',

    # my info
    author='Nicholas Carrara',
    author_email='nmcarrara.physics@gmail.com',

    # where to find the source
    url='https://github.com/infophysics/BlipNet',

    # requirements
    install_reqs=[],

    # packages
    # package_dir={'':'Blipnet'},
    packages=find_packages(
        # 'Blipnet',
        exclude=['tests'],
    ),
    include_package_data=True,
    package_data={'': ['*.yaml']},

    # classifiers
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Experimental Physics',
        'License :: GNU',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
    ],
    python_requires='>3.7',

    # possible entry point
    entry_points={
        'console_scripts': [
            'blipnet = blipnet.programs.run_blipnet:run',
        ],
    },
)
