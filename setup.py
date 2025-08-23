from setuptools import setup, find_packages

setup(
    name="MAS",
    version="0.1.0",
    description="Multi-Agent System for LLMs",
    author="Maojia",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[],  # 可根据 requirements.txt 填写依赖
    python_requires=">=3.10",
)
