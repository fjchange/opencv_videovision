import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="torchvideotransforms",
    version="0.0.1",
    author="Jia-Chang Feng",
    author_email="fjchange@hotmail.com",
    description="Opencv based Data augmentation for videos as stack of images for PyTorch",
    download_url="https://github.com/fjchange/videotransforms_opencv/archive/0.0.1.tar.gz",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fjchange/videotransforms_opencv",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU GENERAL PUBLIC LICENSE",
        "Operating System :: OS Independent",
        "Development Status :: 2 - Pre-Alpha",
    ],
    install_requires=[
        'torch',
        'opencv-python',
    ],
    python_requires='>=3.6',
)