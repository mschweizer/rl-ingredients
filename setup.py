from setuptools import setup, find_packages

setup(name='rl-ingredients',
      version='0.1',
      description='Provides common sacred ingredients for reinforcement learning experiments.',
      url='https://github.com/mschweizer/rl-ingredients',
      author='Marvin Schweizer',
      author_email='schweizer@kit.edu',
      license='MIT',
      packages=find_packages(),
      install_requires=[
          'sacred==0.8.4',
          'gym==0.21',
          'randomname==0.2.1',
          'stable-baselines3==1.8.0',
          'PyYAML==6.0',
          'tensorboard==2.12.0'
      ],
      include_package_data=True,
      python_requires='>=3.7',
      )
