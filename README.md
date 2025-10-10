# Diabetes Risk Prediction App

A machine learning application that predicts diabetes risk based on health parameters using Streamlit.

## 🚀 Features

- **Interactive Web Interface**: Easy-to-use Streamlit dashboard
- **Real-time Predictions**: Instant diabetes risk assessment
- **Health Parameter Input**: Comprehensive health data collection
- **Risk Visualization**: Clear high/low risk indicators
- **Model Export**: Exported trained model for deployment

## 📋 Prerequisites

- Python 3.8+
- Required packages (see requirements.txt)

## 🛠️ Installation

1. **Clone or download the project files**

2. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Jupyter notebook first:**
   - Open `Untitled.ipynb`
   - Run all cells to train the model and export files
   - This will create the necessary model files:
     - `best_diabetes_model.pkl`
     - `diabetes_scaler.pkl`
     - `feature_names.pkl`

4. **Launch the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

## 📊 How to Use

1. **Open the app** in your browser (usually http://localhost:8501)

2. **Input health parameters** using the sidebar sliders:
   - Pregnancies
   - Glucose level
   - Blood pressure
   - Skin thickness
   - Insulin level
   - BMI
   - Diabetes pedigree function
   - Age

3. **Click "Predict Diabetes Risk"** to get instant results

4. **View the prediction**:
   - **High Risk**: Red warning with medical consultation recommendation
   - **Low Risk**: Green confirmation with healthy lifestyle advice

## 🏗️ Project Structure

```
├── Untitled.ipynb              # Main analysis notebook
├── app.py                      # Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── dataset-diabete-*.csv       # Diabetes dataset
├── best_diabetes_model.pkl     # Exported model (created after running notebook)
├── diabetes_scaler.pkl         # Exported scaler (created after running notebook)
└── feature_names.pkl           # Feature names (created after running notebook)
```

## 🔬 Model Details

- **Algorithm**: Optimized machine learning pipeline
- **Preprocessing**: KNN imputation, outlier removal, standardization
- **Clustering**: K-means for risk categorization
- **Hyperparameter Tuning**: GridSearchCV optimization
- **Performance**: High accuracy diabetes risk prediction

## ⚠️ Disclaimer

This application is for **educational and research purposes only**. It is not intended to replace professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical decisions.

## 🎯 Features of the App

- **Responsive Design**: Works on desktop and mobile
- **Real-time Updates**: Instant predictions as you adjust parameters
- **Visual Feedback**: Color-coded risk indicators
- **Professional UI**: Clean, medical-themed interface
- **Confidence Scores**: Shows prediction confidence when available

## 🚀 Deployment

To deploy this app:

1. **Local deployment**: Run `streamlit run app.py`
2. **Cloud deployment**: Use platforms like Streamlit Cloud, Heroku, or AWS
3. **Docker deployment**: Create a Dockerfile for containerized deployment

## 📈 Model Performance

The model achieves high accuracy through:
- Comprehensive data preprocessing
- Advanced hyperparameter optimization
- Multiple algorithm comparison
- Cross-validation for robust evaluation

## 🤝 Contributing

Feel free to contribute by:
- Improving the UI/UX
- Adding new features
- Optimizing the model
- Enhancing documentation

## 📄 License

This project is open source and available under the MIT License.
