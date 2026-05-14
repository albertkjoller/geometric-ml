from setuptools import find_packages, setup
from pathlib import Path

# Read requirements from a requirements.txt file
requirements_path = Path(__file__).parent / "requirements.txt"
with requirements_path.open() as f:
    requirements = f.read().splitlines()

setup(
    name='geomtools',
    packages=find_packages(),
    version='1.0.0',
    description='Toolbox for geometric mechanics',
    author='Albert Kjøller Jacobsen',
    install_requires=requirements,
)