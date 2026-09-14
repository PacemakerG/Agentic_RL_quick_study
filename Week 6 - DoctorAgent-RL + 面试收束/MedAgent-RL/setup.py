from setuptools import find_namespace_packages, setup

setup(
    name='medagent-rl',
    version='0.1.0',
    package_dir={'': '.'},
    packages=find_namespace_packages(include=['ragen', 'ragen.*']),
    author='Yulong Ge',
    description='Multi-turn medical consultation policy training with SFT and GRPO',
    install_requires=[], 
    package_data={'ragen': ['*/*.md']},
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
    ]
)
