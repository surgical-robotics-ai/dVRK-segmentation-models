import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="surg_seg",
    version="0.0.0",
    author="Juan Antonio Barragan",
    author_email="jbarrag3@jh.edu",
    description="Surgical segmentation models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    install_requires=["numpy", "rich", "click"],
    include_package_data=True,
    python_requires=">=3.8",
    # scripts=[ #This does not get uninstalled with `pip uninstall surg_seg`
    #     "surg_seg/Scripts/RosVideoRecord/ros_video_record.py",
    #     "surg_seg/Scripts/generate_labels.py",
    # ],
    entry_points={
        "console_scripts": [
            "surg_seg_generate_labels = surg_seg.Scripts.generate_labels:main",
            "surg_seg_ros_video_record = surg_seg.Scripts.RosVideoRecord.ros_video_record:main",
        ]
    },
)
