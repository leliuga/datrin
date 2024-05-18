from setuptools import find_packages, setup


def get_requirements(path: str):
    return [l.strip() for l in open(path)]


setup(
    name="daTrin",
    version="0.0.1",
    description="datrin is a dataset train and inference learning material",
    author="Evaldas Leliuga",
    author_email="labas@k8s.lt",
    license="MPL-2.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=get_requirements("requirements.txt"),
)
