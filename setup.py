from setuptools import setup, find_packages


setup(
    name="cherab-mastu",
    version="1.1.0",
    license="EUPL 1.1",
    namespace_packages=['cherab'],
    packages=find_packages(),
    install_requires=['cherab'],
    include_package_data=True,
)
