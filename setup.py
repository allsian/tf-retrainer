from distutils.core import setup

setup(name='tf_retrainer',
      version='1.0',
      description='transfer learning in tensorflow',
      long_description=open('README.md').read(),
      author='Chris Messier',
      license='BSD 3-Clause',
      author_email='messiercr@gmail.com',
      url='https://github.com/messiest/tf-retrainer',
      packages=['tf_retrainer', 'tf_retrainer.utils', 'tf_retrainer.image_retraining'],
      classifiers=['Development Status :: 1 - Planning',
                   'Intended Audience :: Developers',
                   'License :: OSI Approved :: BSD License',
                   'Operating System :: OS Independent',
                   'Programming Language :: Python',
                   'Programming Language :: Python :: 3',
                   'Topic :: Scientific/Engineering :: Artificial Intelligence'],
      )