#!/bin/sh
#touch docs/_build/html/.nojekyll
#cd $TRAVIS_BUILD_DIR/docs
#make  html
#cd $TRAVIS_BUILD_DIR
make -C docs/ html