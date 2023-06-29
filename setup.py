from setuptools import setup, find_packages

setup(
    name="tggnn",
    version="0.3.0",
    packages=find_packages(include=["tggnn*"]),
    package_data={"tggnn": ["py.typed"]},
)
