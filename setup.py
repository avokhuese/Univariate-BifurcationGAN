from setuptools import setup, find_packages


classifiers = [
    'Development Status :: 4 - Beta',   
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.8',
]

setup(
    name='Univariate Time Series Data Augmentation',
    version='0.0.2',
    description='GAN time series data augmentation with several GAN variates and novel BifurcationGAN models,
    long_description=open('README.txt').read(),
    url='',
    author='Alexander Victor',
    author_email='alexander.victor@dbs.ie'
l   icense='MIT',
    classifiers=classifiers,
    packages=find_packages(),
    install_requires=[''],
    python_requires='>=3.8',
    include_package_data=True,
    zip_safe=False,
    keywords='calculator basic math',
)