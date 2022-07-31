#!/usr/bin/env python
from setuptools import setup

REQUIREMENTS = []
with open('requirements.txt') as requirements_file:
    REQUIREMENTS.extend(requirements_file.readlines())

# dev requirements
with open('requirements-dev.txt') as requirements_file:
    REQUIREMENTS.extend(filter(lambda line: not line.strip().startswith("-r"), requirements_file.readlines()))

setup (
    name='mouse-pointer-controller',
    description='An OpenVino Gaze estimation model based mouse pointer controller',
    version='0.1.0',
    packages=['mouse_pointer_controller'],
    author='Wajih Ouertani',
    author_email='wajih.ouertani@gmail.com',
    install_requires=REQUIREMENTS,
    scripts=["scripts/create-env.sh", "scripts/download_models.sh", "mouse_pointer_controller/mouse_controller.py"],
    # package_data={
    # }
)