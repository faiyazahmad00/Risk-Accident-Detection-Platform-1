# Risk Accident Detection Platform

A beginner-friendly, software-only demonstration of a machine-learning dashboard
that estimates an illustrative road-condition risk score.

## Important limitation

This starter project creates **synthetic (artificial) training data** so that it
can run without downloading a dataset. The displayed score is not a real-world
accident probability, is not validated for road safety, and must not be used for
driving, emergency, insurance, or traffic-control decisions.

For a research-quality project, replace the synthetic data with a suitable,
documented real dataset and perform careful validation.

## 1. Project files

- `app.py` — Streamlit dashboard and demonstration ML model.
- `requirements.txt` — Python libraries needed by the project.
- `README.md` — Setup and usage instructions.

## 2. Software requirements

- Python 3.9 or newer
- Visual Studio Code (recommended)
- Internet connection for the first-time library installation
- A modern web browser

No Arduino, sensors, cameras, or other physical hardware are required.

## 3. Install and run

Open the project folder in VS Code. Open its terminal and run:

### Windows

```bash
python -m venv venv
venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run app.py
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Streamlit will print a local address, usually `http://localhost:8501`.
Open that address in your browser.

## 4. How to use

1. Start the app with `streamlit run app.py`.
2. Select weather, road surface, lighting and traffic density.
3. Set the hour, speed-limit context and road curvature.
4. Read the demonstration score and its Low, Moderate or High category.
5. Expand the sections to see the selected inputs and synthetic-data evaluation.

## 5. Technology used

- Python — main programming language
- Pandas and NumPy — data handling and numerical operations
- Scikit-learn — Random Forest regression model
- Streamlit — interactive browser-based dashboard

## 6. How the code works

1. `create_demo_data()` generates artificial example records.
2. `train_demo_model()` prepares the features and trains a Random Forest model.
3. The sidebar collects user-selected road and environment conditions.
4. The app encodes the inputs in the same feature format used for training.
5. The model predicts a score and the app displays a category.

## 7. Troubleshooting

- **`python` is not recognised:** Install Python and enable the option to add it
  to PATH, then reopen the terminal.
- **`streamlit` is not recognised:** Activate the virtual environment and run
  `python -m pip install -r requirements.txt`.
- **Port 8501 is busy:** Run `streamlit run app.py --server.port 8502`.
- **Package installation fails:** Check your internet connection and Python
  version, then retry the installation.

## 8. Future improvements

- Use a trustworthy, documented road-accident dataset.
- Add data cleaning, feature engineering and separate model-training scripts.
- Compare suitable models using train/validation/test splits.
- Report appropriate metrics and limitations.
- Add explainability and carefully validated uncertainty estimates.
- Keep the dashboard clearly labelled as decision support, not a substitute
  for road-safety judgment.

## 9. Academic note

This is a starter implementation aligned with the software-only concept in the
project report. It is intended for learning and demonstration. The synthetic
data and sample evaluation results should not be presented as actual accident
statistics or evidence of real-world model accuracy.
