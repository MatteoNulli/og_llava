from setuptools import setup, find_packages

# Read the content of the README file
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

# Read the content of the requirements.txt file
with open('requirement.txt', 'r', encoding='utf-8') as f:
    requirements = f.read().splitlines()

setup(
    name="llava",
    version="0.1.0",
    packages=find_packages(include=['llava', 'llava.*']),
    python_requires='>=3.8',
    install_requires=requirements,
)