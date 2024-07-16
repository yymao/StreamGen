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
        'random',
        'pandas',
        'glob'
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
