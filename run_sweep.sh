#!/bin/bash
# Run the tuned lens sweep with full-vocab and more sequences

cd /home/chiragr2/computational-mechanics/geometric_interpretability_LLMs

echo "Starting tuned lens sweep (50 sequences, full-vocab)..."
python -m experiments.tuned_lens_per_layer experiments/configs/tuned_lens_sweep.yaml

echo "Done! Results in results/tuned_lens_sweep_more_sequences/"
