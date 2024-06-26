import os
from glob import glob
from setuptools import setup

package_name = 'graph_matching'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),glob(os.path.join('launch', '*.launch.py'))),
        (os.path.join('share', package_name, 'config'),glob(os.path.join('config', '*.yaml'))),
        (os.path.join('share', package_name, 'config'),glob(os.path.join('config', '*.json'))),

    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Jose Andres Millan Romera',
    maintainer_email='josmilrom@gmail.com',
    description='Graph matching tool for S-graphs',
    license='MIT liscense',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'graph_matching = graph_matching.graph_matching_node:main'
        ],
    },
)
