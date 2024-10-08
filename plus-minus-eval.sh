# Run adversarial accuracy evaluation for plus-minus dataset
echo "Running plus-minus dataset evaluation"

cd pipeline/aml
python analysis.py plus-minus pqc

echo "Finished evaluation"