from setuptools import setup, find_packages

def readme():
    with open('README.md') as f:
            return f.read()

setup(name='scitorch',
      version='0.0.1',
      description='Scientific computing based on PyTorch.',
      long_description=readme(),
      classifiers=[
          'Development Status :: 2 - Pre-Alpha',
          'License :: OSI Approved :: Apache Software License',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Topic :: Scientific/Engineering :: Physics',
          'Topic :: Scientific/Engineering :: Mathematics',
      ],
      url='http://github.com/christophfritz/scitorch',
      author='Christoph Fritz',
      author_email='christophfritz95@icloud.com',
      license='Apache License 2.0',
      packages=find_packages(),
      setup_require=[
          'pytest-runner'
      ],
      test_require=[
          'pytest', 
      ],
      install_requires=[
          'torch',
          'torchvision',
          'pytest',
          'virtualenvwrapper',
      ],
      zip_safe=False)
