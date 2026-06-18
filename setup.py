from pathlib import Path
from setuptools import setup

description = ['Training and evaluation of the ECCV 2026 paper PolyLayout: Multi-room Manhattan Layout Estimation']

with open(str(Path(__file__).parent / 'README.md'), 'r', encoding='utf-8') as f:
    readme = f.read()

with open(str(Path(__file__).parent / 'requirements.txt'), 'r') as f:
    dependencies = f.read().split('\n')

extra_dependencies = ['jupyter', 'scikit-learn', 'ffmpeg-python', 'kornia', 'rerun-sdk', 'rerun-sdk[notebook]', 'scikit-learn']

deeplsd_dependencies = ['flow_vis', 'kornia>=0.6', 'scikit-image', 'seaborn', 'deeplsd @ git+https://github.com/cvg/DeepLSD.git', 'pytlsd @ git+https://github.com/iago-suarez/pytlsd.git']

setup(
    name='PolyLayout',
    version='1.0',
    packages=['pixloc'],
    python_requires='>=3.6',
    install_requires=dependencies,
    extras_require={'extra': extra_dependencies, 'deeplsd': deeplsd_dependencies},
    author='Paul-Edouard Sarlin (PixLoc), Gustav Hanning (PolyLayout)',
    description=description,
    long_description=readme,
    long_description_content_type="text/markdown",
    url='https://github.com/ghanning/PolyLayout/',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
