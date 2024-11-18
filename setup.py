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
        exclude = ['discretizer', 'engines', 'models', 'solvers', 'docs', 'thirdparty']),

    # Now only include already built libraries, and build_info, otherwise it will not find the file when using darts.
    package_data={'darts': ['*.pyd', '*.so', '*.dll', 'build_info.txt']},

    # Package metadata
    description='Delft Advanced Research Terra Simulator',

    # handle correct platform wheel names
    distclass=BinaryDistribution
)
