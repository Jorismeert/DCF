# DCF


Enviroment:
# Create environment for stock analysis
python -m venv stock_analysis
source stock_analysis/bin/activate

# Install needed packages
pip install yfinance pandas numpy matplotlib jupyter ipykernel

# Make it available in Jupyter
python -m ipykernel install --user --name=stock_analysis --display-name="Stock Analysis"

# Save environment
pip freeze > requirements.txt

# Exit enviroment
deactivate
