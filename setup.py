
import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='opencv_videovision',
    version='0.0.1',
    author='Jia-Chang Feng',
    author_email='fjchange@hotmail.com',
    description='A drop-in replacement for Torchvision Video Transforms using OpenCV',
    keywords='pytorch image augmentations',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/fjchange/opencv_videovision',
    packages=['opencv_videovision'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU GENERAL PUBLIC LICENSE",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
