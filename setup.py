from setuptools import find_packages, setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()
    
setup(
    name="fighingcv",
    version="1.0.0",
    author="Cena Av",
    author_email="sinaavakh@gmail.com",
    description=(
        "FightingCV Codebase For Attention,Backbone, MLP, Re-parameter, Convolution"
    ),
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords=(
        "Attention"
        "Backbone"
    ),
    license="Apache",
    url="https://github.com/cenaav/External-Attention-pytorch",
    package_dir={"": "model"},
    packages=find_packages("model"),
    
    # entry_points={
    #     "console_scripts": [
    #         "huggingface-cli=huggingface_hub.commands.huggingface_cli:main"
    #     ]
    # },
    
    python_requires=">=3.8.0",
    install_requires=requirements,
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    project_urls={
        "Forked From": "https://github.com/xmu-xiaoma666/External-Attention-pytorch"
    }
)