from setuptools import setup


def readme():
    with open('README.md') as f:
        return f.read()

setup(
    name='sequence-classification',
    version='0.1',
    description='module for comparing sequences classifiers',
    author='Berezowski Jakub, Lipowska Magda,Miara Piotr, Szczepaniak Grzegorz',
    author_email='mail',
    license='MIT',
    zip_safe=False,
    setup_requires=['pytest-runner'],
    install_requires=[],
    tests_require=['pytest']
)
