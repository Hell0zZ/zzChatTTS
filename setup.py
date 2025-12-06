from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="zzChatTTS",
    version="0.1.0",
    author="ZZ",
    description="A ChatTTS text-to-speech implementation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Hell0zZ/zzChatTTS",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "ChatTTS>=0.1.0",
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "soundfile>=0.12.0",
    ],
)
