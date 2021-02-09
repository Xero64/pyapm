import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyapm",
    version="0.0.2",
    author="Xero64",
    author_email="xero64@gmail.com",
    description="Aerodynamic Panel Method in Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Xero64/pyapm",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    entry_points = {
        'console_scripts': ['pyapm=pyapm.__main__:main',],
    }
)
