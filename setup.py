from distutils.core import setup

setup(
    name='biotmpy',
    packages=['biotmpy'],
    version='0.1',
    license='MIT',
    description='BIOmedical Text Mining with PYthon',
    author='Nuno Alves',
    author_email='n4lv3s@gmail.com',
    url='https://github.com/BioSystemsUM/biotmpy',
    download_url='https://github.com/user/reponame/archive/v_01.tar.gz',
    keywords=['SOME', 'MEANINGFULL', 'KEYWORDS'],
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'scikit-learn',
        'nltk',
        'gensim'
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9'
        'Programming Language :: Python :: 3.10'
    ],
)
