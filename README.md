# 🛠️ Industrial Data Processing Platform

A web-based platform for processing, analyzing, and correcting industrial sensor data. This application provides a user-friendly interface for data validation, cleaning, and automatic error correction.

## 🌟 Key Features

### 1. User Authentication
- Secure login and registration system
- User identification format: name-company
- Session management for secure access

### 2. Data Upload & Validation
- CSV file upload with format validation
- Required columns:
  - `date` column (format: YYYY-MM-DD HH:MM:SS)
  - At least one numeric column
- Automatic file type verification

### 3. Data Processing Pipeline
1. **Initial Filtering**
   - Data sorting by timestamp
   - Selection of relevant variables
   - Output: `filtered_data.csv`

2. **Missing Data Handling**
   - Automatic frequency detection
   - Missing timestamp identification
   - Value interpolation using temporal neighbors
   - Outputs:
     - `complete_data.csv`
     - `missing_dates.txt`

3. **Anomaly Detection & Correction**
   - Detection of value deviations (>5× average variation)
   - Negative value correction (configurable per variable)
   - Correlation-based inconsistency detection
   - Outputs:
     - `corrected_data.csv`
     - `detected_failures.csv`

### 4. Data Storage
- User credentials: `users.txt`
- Negative variable configuration: `negative_variables.txt`
- All processed data files in `upload/` directory

## 📋 Requirements

### Software
- Python 3.8 or higher
- Flask web framework
- Pandas for data manipulation
- NumPy for numerical operations

### Dependencies
```bash
pip install -r requirements.txt
```

## 🚀 Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the application:
   ```bash
   python app.py
   ```

3. Access the platform:
   - Open your browser
   - Navigate to `http://localhost:10000`

## 📊 Output Files Description

| File | Description |
|------|-------------|
| `filtered_data.csv` | Initial cleaned dataset with selected variables |
| `complete_data.csv` | Dataset with completed timestamps and interpolated values |
| `missing_dates.txt` | List of timestamps where data was interpolated |
| `corrected_data.csv` | Final dataset after all corrections |
| `detected_failures.csv` | Detailed report of detected and corrected anomalies |

## 🔒 Security Features

- Secure password storage
- Session-based authentication
- Protected routes requiring login
- Secure file handling

## 🎨 User Interface

- Modern dark theme
- Responsive design
- Intuitive navigation
- Clear error messages and feedback
- Progress tracking for data processing

## 🔄 Data Processing Flow

1. User authentication
2. CSV file upload
3. Variable selection
4. Data filtering and cleaning
5. Missing data completion
6. Anomaly detection and correction
7. Results download

## 📝 Notes

- All dates must be in YYYY-MM-DD HH:MM:SS format
- Maximum file size: 100MB
- Supported file format: CSV only
- Automatic backup of processed files