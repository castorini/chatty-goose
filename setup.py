import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name="chatty-goose",
    version="0.1.0",
    author="Anserini Gaggle",
    author_email="anserini.gaggle@gmail.com",
    description="A conversational passage retrieval toolkit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jacklin64/Conversational-IR",
    install_requires=requirements,
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
