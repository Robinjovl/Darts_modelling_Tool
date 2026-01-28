from setuptools import setup, find_packages, Distribution

# custom class to inform setuptools about self-compiled extensions in the distribution
# and hence enforce it to create platform wheel
class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True

setup(distclass=BinaryDistribution)
