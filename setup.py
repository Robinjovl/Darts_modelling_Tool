from setuptools import setup, find_packages, Distribution

# custom class to inform setuptools about self-compiled extensions in the distribution
# and hence enforce it to create platform wheel
class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True

setup(
    # Add packages that are inside folder darts-package
    packages = find_packages(
    	where = '.',
        exclude = ['discretizer', 'engines', 'models', 'solvers', 'docs', 'thirdparty']),

    # Now only include already built libraries, and build_info, otherwise it will not find the file when using darts.
    package_data={'darts': ['*.pyd', '*.so', '*.dll', 'CHANGELOG.md', 'build_info.txt', 'libstdc++.so.6']},

    # Add darts command line interface
    entry_points = {
        'console_scripts': ['darts = darts.tools.cli:main',],
    },

    # Package metadata
    description='Delft Advanced Research Terra Simulator',

    # handle correct platform wheel names
    distclass=BinaryDistribution
)
