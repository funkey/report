from setuptools import setup

setup(
        name='report',
        version='0.2',
        description='Simple report plotting library',
        author='Jan Funke',
        author_email='jfunke@iri.upc.edu',
        url='https://github.com/funkey/report',
        packages=['report'],
        requires=['bokeh', 'pandas'],
)
