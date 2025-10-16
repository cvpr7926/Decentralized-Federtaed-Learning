# 1.Prerequisites

Install PyTorch C++ (LibTorch)

wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcpu.zip

unzip libtorch-cxx11-abi-shared-with-deps-2.1.0+cpu.zip

# 2.Build Commands

mkdir build

cd build

cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..

cmake --build . --config Release

cd..

# 3.Download CIFAR-10 Dataset

mkdir -p data

cd data

Download CIFAR-10 binary version

wget https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz

tar -xzf cifar-10-binary.tar.gz

cd ..

# 4.Run the Program

./build/federated_learning

