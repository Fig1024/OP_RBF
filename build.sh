#!/bin/bash

ROOT_DIR=${PWD}
BUILD_DIR=${ROOT_DIR}/build/

rm -rf ${BUILD_DIR}
mkdir -p ${BUILD_DIR} && cd ${BUILD_DIR}

cmake ..
make
echo "make complete"
cp OP_RBF ../
echo "copy complete" 
