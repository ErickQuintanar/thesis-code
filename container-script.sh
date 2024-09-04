# Create config files
echo "Creating required configuration files"
cd pipeline/test_configs
python create-configs.py iris
python create-configs.py plus-minus
python create-configs.py mnist10
python create-configs.py mnist4
python create-configs.py mnist2

cd ..

echo "Training models according to config files for Iris dataset"
sh run_tests.sh iris # testing if it runs
echo "Training models according to config files for Plus-Minus dataset"
sh run_tests.sh plus-minus
echo "Training models according to config files for MNIST10 dataset"
sh run_tests.sh mnist10
echo "Training models according to config files for MNIST4 dataset"
sh run_tests.sh mnist4
echo "Training models according to config files for MNIST2 dataset"
sh run_tests.sh mnist2

echo "Finished training"