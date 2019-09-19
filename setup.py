from setuptools import setup, find_packages

setup_requires = []

install_requires = [
    # 'django==1.6b4',
    'numpy',
    'pandas',
    'keras',
    'tensorlow'
]

dependency_links = [
    # 'git+https://github.com/django/django.git@stable/1.6.x#egg=Django-1.6b4',
]

print('Translator modules\n%s' % install_requires)

setup(name='Translator',
      version='1.0',
      description='Translator services',
      author='Kioryu',
      author_email='1218Moonlight@gmail.com',
      packages=find_packages(),
      install_requires=install_requires,
      setup_requires=setup_requires,
      dependency_links=dependency_links,
      scripts=[],
      entry_points={}
      )
