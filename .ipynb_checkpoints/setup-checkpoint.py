from setuptools import setup, find_packages

setup(
    name='StreamGen',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'astropy==5.1',
        'scipy==1.10.0',
        'numpy==1.24.3',
        'matplotlib==3.2.2',
        'pandas',
        'tqdm',
        'ipykernel',
        'jupyterlab',
        # Additional required packages
        'aiohttp==3.9.4',
        'attrs',
        'codecov==2.1.12',
        'requests',
        'click',
        'future',
        'nose',
        'cython',
        'beautifulsoup4',
        'zipp',
        'tornado',
        'jsonschema',
        'sympy',
        'fsspec',
        'werkzeug',
        'wrapt',
        'jinja2',
        'threadpoolctl',
        'idna',
        'html5lib',
        'keyring',
        'numba',
        # Specific versions required by tensorflow and virtualenv
        'typing-extensions==4.5.0',
        'platformdirs<3',
        'traitlets>=5.3.0'
        # If additional packages are needed, they can be added here
    ],
    entry_points={
        'console_scripts': [
            # If you have scripts to run, specify them here
            # 'myscript = mypackage.myscript:main',
        ],
    },
    author='Adriana Dropulic',
    author_email='dropulic@princeton.edu',
    description='StreamGen, arxiv:2407.XXXX',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/adropulic/StreamGen',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8.3',
)
