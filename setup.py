from setuptools import setup, find_packages


def get_install_requires() -> str:
    return [
        'gym>=0.21.0',
        'numpy',
    ]
    
def get_extras_require() -> str:
    req = {
        "metadrive":
        ["metadrive-simulator@git+https://github.com/HenryLHH/metadrive_clean.git@main"],
    }
    return req

setup(name='fusion',
      packages=["fusion"],
      description="FUSION: Library Zoo for offline Safe RL in Autonomous Driving",
      long_description=open("README.md", encoding="utf8").read(),
      long_description_content_type="text/markdown",
      url="https://github.com/HenryLHH/fusion.git",
      author="FUSION contributors",
      author_email="haohongl@andrew.cmu.edu",
      license="Apache",
      include_package_data=True,
      version='1.1.0',
      install_requires=get_install_requires(),
      extras_require=get_extras_require(),
      python_requires='>=3.8.0',
)


