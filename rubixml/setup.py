from setuptools import setup, find_packages
from rubixml import VERSION, AUTHOR

setup(name='rubixml',
      version=VERSION,
      description='A Simple Machine Learning Libraries',
      url='https://github.com/yinchuandong/sentiment-analysis',
      author=AUTHOR,
      license='MIT',
      packages=find_packages(exclude=('tests',
                                      'test_*',
                                      '__pycache__'
                                      'example_data',
                                      '.data',
                                      '.textcnn',
                                      '.vector_cache',
                                      '.pytest_cahce')),
      install_requires=[
          'torch',
          'torchtext',
          'numpy',
          'sklearn',
          'pandas',
      ],
      zip_safe=True)
