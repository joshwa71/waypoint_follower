from setuptools import find_packages, setup

package_name = 'waypoint_follower'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='jiaojh1994@gmail.com',
    description='Waypoint follower node with PD control and dynamic waypoint subscription',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'waypoint_follower = waypoint_follower.waypoint_follower:main',
        ],
    },
)
