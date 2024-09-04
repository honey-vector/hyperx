from setuptools import setup, find_packages

setup(
    name='hypercomplex-algebra',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'tensorflow>=2.0.0',
    ],
    extras_require={
        'torch': ['torch'],
    },
    author='Renato Boemer',
    author_email='boemer00@gmail.com',
    description='A library for general hypercomplex algebras with Keras integration',
    license='MIT',
    url='https://github.com/honey-vector/hyperx',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
