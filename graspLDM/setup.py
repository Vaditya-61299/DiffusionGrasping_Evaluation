import os

from setuptools import find_namespace_packages, find_packages, setup

from grasp_ldm import __version__

# here = os.path.abspath(os.path.dirname(__file__))
# requires_list = []
# with open(os.path.join(here, 'requirements.txt'), encoding='utf-8') as f:
#     for line in f:
#         requires_list.append(str(line))

setup(
    name="grasp_ldm",
    version=__version__,
    author="Kuldeep Barad",
    # TODO: Improve grasp_ldm_utils module by combining internal and external utils
    packages=["grasp_ldm", "grasp_ldm.tools", "grasp_ldm_utils","isaac_evaluation","se3dif"],
    # packages=find_packages(),
    package_dir={
        "grasp_ldm": "grasp_ldm",
        "grasp_ldm.tools": "/home/aditya/Desktop/part_2/graspLDM/tools",
        "grasp_ldm_utils": "/home/aditya/Desktop/part_2/graspLDM/grasp_ldm/utils",
        "isaac_evaluation": "/home/aditya/Desktop/part_2/graspLDM/isaac_evaluation",
        "se3dif": "/home/aditya/Desktop/part_2/graspLDM/se3dif"
    },
    python_requires=">=3.8.0, <3.10",
    # install_requires=requires_list,
)
