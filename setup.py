from setuptools import setup

setup(name='powergrid_data_generation',
      version='0.1',
      description='Some utilities to generate data',
      long_description='Data are generated based on the French total consumption of 2012.',
      classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3.x',
      ],
      url='',
      author='Benjamin DONNOT',
      author_email='benjamin.donnot@rte-france.com',
      license='GPL-v3',
      packages=['pgdg'],
      install_requires=["numpy", "pandas", "pandapower", "Grid2Op"],
      include_package_data=False,
      zip_safe=False)