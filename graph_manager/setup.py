from setuptools import setup

package_name = 'graph_manager'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Jose Andres Millan Romera',
    maintainer_email='josmilrom@gmail.com',
    description='Graph manager tool for Sgraphs',
    license='MIT liscense',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'graph_manager = graph_manager.graph_manager:main',
        ],
    },
)
