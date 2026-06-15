#!/bin/bash
echo "Setting up local editable environment..."
pip install -e .
pip install -e ../ml-switcheroo-compiler
echo "Done!"
