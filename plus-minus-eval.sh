# Run adversarial accuracy evaluation for plus-minus dataset
echo "Running plus-minus dataset evaluation"

cd pipeline/aml
python plus-minus-reduced-eval.py

echo "Finished evaluation"