import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyro_agents",
    version="0.0.1",
    author="Robert Ness",
    author_email="robertness@gmail.com",
    description="Agent models implemented in Pyro",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/robertness/pyro_agents",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
