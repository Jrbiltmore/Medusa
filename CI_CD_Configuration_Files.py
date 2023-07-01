import os

# CI/CD Framework Files
circle_ci_yml = """
version: 2
jobs:
  build:
    working_directory: ~/
    docker:
      - image: cimg/node:14.8.0
    steps:
      - checkout
      - run:
          name: Install Dependencies
          command: |
            npm install
      - run:
          name: Build Application
          command: |
            npm run build
      - run:
          name: Test Application
          command: |
            npm test
"""

travis_yml = """
dist: trusty
language: node_js
node_js:
  - "node"

before_install:
  - npm install

script:
  - npm run build
  - npm test

after_success:
  - npm run deploy
"""

gitlab_ci_yml = """
stages:
  - test
  - deploy

test:
  stage: test
  image: cimg/node:14.8.0
  script:
    - npm install
    - npm run build
    - npm test
  coverage: '/coverage/'
  artifacts:
    when: always
    paths:
      - ./built

deploy:
  stage: deploy
  script: npm run deploy
"""

# Create configuration files
if not os.path.exists('ci'):
    os.makedirs('ci')

with open(os.path.join('ci', 'circle.yml'), 'w') as cf:
    cf.write(circle_ci_yml)

with open(os.path.join('ci', 'travis.yml'), 'w') as cf:
    cf.write(travis_yml)

with open(os.path.join('ci', 'gitlab.ci.yml'), 'w') as cf:
    cf.write(gitlab_ci_yml)

print('CI/CD Configuration Files successfully created!')

