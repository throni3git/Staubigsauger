from setuptools import setup, find_packages

try:
    long_description = open("README.md").read()
except IOError:
    long_description = ""

setup(
    name="Staubigsauger",
    version="0.1.0",
    description="Audio file cleanup with spectrogram display",
    license="MIT",
    author="Thomas Thron",
    packages=find_packages(),
    install_requires=[
        "autopep8",
        "numpy",
        "matplotlib",
        "SoundFile",
        "scipy",
        "sounddevice",
        "pylint",
        "numba",
        "PySide2",
    ],
    long_description=long_description,
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
    ],
    python_requires=">=3.6",
)
