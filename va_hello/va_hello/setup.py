from setuptools import setup

package_name = 'va_hello'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Grupo 00',
    maintainer_email='grupo00@ua.pt',
    description='Primeira app de VA',
    license='MIT',
    entry_points={
        'console_scripts': [
            'fake_scan = va_hello.fake_scan:main',
            'min_distance = va_hello.min_distance:main',
        ],
    },
)
