from setuptools import setup, find_packages, Distribution

# custom class to inform setuptools about self-compiled extensions in the distribution
# and hence enforce it to create platform wheel
class BinaryDistribution(Distribution):
    def has_ext_modules(foo):
        return True

setup(
    # Add packages that are inside folder darts-package
    packages = find_packages(
	where = '.',
	include = ['darts']),

    # Now only include already built libraries
    package_data={'darts': ['*.pyd', '*.so', '*.dll']},

    # Package metadata
    description='Delft Advanced Research Terra Simulator',

    # handle correct platform wheel names
    distclass=BinaryDistribution,
)
