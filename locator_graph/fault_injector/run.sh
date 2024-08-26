#!/bin/bash

for i in {1..3}; do
    echo "Running iteration $i..."
    python script.py --model 'GRU7' --operator 'backbone_activation' --logs_dir '/content/drive/MyDrive/models_faults/'
    echo "------------------------------"  # Visual separator for clarity
done
