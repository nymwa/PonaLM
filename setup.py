import setuptools

setuptools.setup(
        name = 'ponalm',
        packages = setuptools.find_packages(),
        install_requires=[
            'numpy', 'torch', 'tqdm'],
        entry_points = {
            'console_scripts':['ponalm = ponalm.cli.ponalm:main']})

