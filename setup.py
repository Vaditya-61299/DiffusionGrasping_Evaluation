import os

from setuptools import find_namespace_packages, find_packages, setup

from graspLDM.grasp_ldm import __version__
from DiffusionFields.se3dif import __version__
from evaluation import __version__

root=os.path.dirname(os.path.abspath(__file__))
print ("root: ", root)

setup(
    name="Diffusion_Evaluation",
    version=__version__,
    author="Aditya Verma",
    # TODO: Improve grasp_ldm_utils module by combining internal and external utils
    packages=["grasp_ldm", "grasp_ldm.tools", "grasp_ldm_utils","evaluation","se3dif"],
    # packages=find_packages(),
    package_dir={
        "grasp_ldm": os.path.join(root,"graspLDM/grasp_ldm"),
        "grasp_ldm.tools": os.path.join(root,"graspLDM/tools"),
        "grasp_ldm_utils": os.path.join(root,"graspLDM/grasp_ldm/utils"),
        "evaluation": os.path.join(root,"evaluation"),
        "se3dif": os.path.join(root,"DiffusionFields/se3dif")
    },
    python_requires=">=3.8.0, <3.10",
)