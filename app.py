import os
import warnings
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, session, flash
import pandas as pd
import numpy as np
from datetime import datetime
from functools import wraps
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.utils import secure_filename

warnings.filterwarnings('ignore')

app = Flask(__name__, 
    static_url_path='',
    static_folder='static')
app.secret_key = 'supersecretkey123'  # Change to a secure key

UPLOAD_FOLDER = 'upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'auth'

class User(UserMixin):
    def __init__(self, id):
        self.id = id

@login_manager.user_loader
def load_user(user_id):
    return User(user_id)

# ========================
# Rutas principales
# ========================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/pricing')
def pricing():
    return render_template('pricing.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        subject = request.form.get('subject')
        message = request.form.get('message')
        
        # Aquí podrías agregar la lógica para enviar el correo electrónico
        # Por ahora solo mostraremos un mensaje de éxito
        flash('¡Gracias por tu mensaje! Te contactaremos pronto.', 'success')
        return redirect(url_for('contact'))
    
    return render_template('contact.html')

# ========================
# Login and authentication
# ========================

USERS_FILE = 'users.txt'

def load_users():
    users = {}
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r') as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    username, password = line.strip().split(':')
                    users[username] = password
    return users

def save_users(users):
    with open(USERS_FILE, 'w') as f:
        for username, password in users.items():
            f.write(f"{username}:{password}\n")

@app.route('/auth', methods=['GET', 'POST'])
def auth():
    users = load_users()

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        action = request.form['action']

        if action == 'login':
            if users.get(username) == password:
                session['user'] = username
                return redirect(url_for('upload'))
            else:
                flash('Usuario o contraseña incorrectos', 'error')

        elif action == 'register':
            if username in users:
                flash('El usuario ya existe', 'error')
            else:
                users[username] = password
                save_users(users)
                flash('Usuario registrado exitosamente. Ahora puedes iniciar sesión.', 'success')

    return render_template('auth.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    flash('You have logged out', 'info')
    return redirect(url_for('auth'))

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('auth'))
        return f(*args, **kwargs)
    return decorated_function

# ========================
# Protected routes
# ========================

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected', 'error')
            return redirect(request.url)
            
        file = request.files['file']
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)

        if not file.filename.endswith('.csv'):
            flash('Please upload a CSV file', 'error')
            return redirect(request.url)

        try:
            # Guardar el archivo con un nombre único
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            unique_filename = f"{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(filepath)
            
            # Guardar el nombre del archivo en la sesión
            session['uploaded_file'] = unique_filename
            
            # Verificar que el archivo sea válido
            df = pd.read_csv(filepath)
            if 'date' not in df.columns:
                flash('CSV file must contain a "date" column', 'error')
                return redirect(request.url)

            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.dropna(subset=['date'])
            
            if len(df) == 0:
                flash('No valid data found in the CSV file', 'error')
                return redirect(request.url)

            numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            if not numeric_columns:
                flash('No numeric columns found in the CSV file', 'error')
                return redirect(request.url)

            flash('File uploaded successfully', 'success')
            return redirect(url_for('form'))

        except Exception as e:
            flash(f'Error processing file: {str(e)}', 'error')
            return redirect(request.url)

    return render_template('upload.html')

@app.route('/form', methods=['GET', 'POST'])
@login_required
def form():
    if 'uploaded_file' not in session:
        flash('Please upload a file first', 'error')
        return redirect(url_for('upload'))
        
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], session['uploaded_file'])
        df = pd.read_csv(filepath)
        
        if 'date' not in df.columns:
            flash('CSV file must contain a "date" column', 'error')
            return redirect(url_for('upload'))

        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
        
        if len(df) == 0:
            flash('No valid data found in the CSV file', 'error')
            return redirect(url_for('upload'))

        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        if not numeric_columns:
            flash('No numeric columns found in the CSV file', 'error')
            return redirect(url_for('upload'))

        min_date = df['date'].min().strftime('%Y-%m-%dT%H:%M:%S')
        max_date = df['date'].max().strftime('%Y-%m-%dT%H:%M:%S')

        return render_template('form.html', variables=numeric_columns, min_date=min_date, max_date=max_date)

    except Exception as e:
        flash(f'Error processing file: {str(e)}', 'error')
        return redirect(url_for('upload'))

@app.route('/process', methods=['POST'])
@login_required
def process():
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'data.csv')
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])

    selected_vars = request.form.getlist('variables')
    negative_vars = request.form.getlist('negative_variables')

    df_filtered = df.sort_values('date')
    df_to_save = df_filtered[['date'] + selected_vars]

    df_to_save_copy = df_to_save.copy()
    df_to_save_copy['date'] = df_to_save_copy['date'].dt.strftime('%Y-%m-%d %H:%M:%S')
    df_to_save_copy.to_csv(os.path.join(app.config['UPLOAD_FOLDER'], 'filtered_data.csv'), index=False)

    # Save negative variables to text file
    with open(os.path.join(app.config['UPLOAD_FOLDER'], 'negative_variables.txt'), 'w') as f:
        for var in negative_vars:
            f.write(f"{var}\n")

    complete_data, missing_dates = complete_missing_data(df_to_save, selected_vars)
    complete_data['date'] = pd.to_datetime(complete_data['date']).dt.strftime('%Y-%m-%d %H:%M:%S')
    complete_data.to_csv(os.path.join(app.config['UPLOAD_FOLDER'], 'complete_data.csv'), index=False)

    with open(os.path.join(app.config['UPLOAD_FOLDER'], 'missing_dates.txt'), 'w') as f:
        for date in missing_dates:
            f.write(f"{date}\n")

    corrected_df, detected_failures = detect_and_correct_failures(complete_data.copy(), selected_vars, negative_vars)
    corrected_df.to_csv(os.path.join(app.config['UPLOAD_FOLDER'], 'corrected_data.csv'), index=False)

    with open(os.path.join(app.config['UPLOAD_FOLDER'], 'detected_failures.csv'), 'w') as f:
        f.write('date,variable,original_value,expected_value,error_type\n')
        for failure in detected_failures:
            f.write(f"{failure['date']},{failure['variable']},{failure['original_value']},{failure['expected_value']},{failure['error_type']}\n")

    return render_template('results.html',
                           selected_vars=selected_vars,
                           negative_vars=negative_vars,
                           start=df['date'].min().strftime('%Y-%m-%d %H:%M:%S'),
                           end=df['date'].max().strftime('%Y-%m-%d %H:%M:%S'))

def complete_missing_data(df_filtered, measurement_columns):
    df_filtered = df_filtered.copy()
    df_filtered['date'] = pd.to_datetime(df_filtered['date'])
    inferred_freq = pd.infer_freq(df_filtered['date']) or '10s'
    date_range = pd.date_range(start=df_filtered['date'].min(), end=df_filtered['date'].max(), freq=inferred_freq)
    complete_dates_df = pd.DataFrame({'date': date_range})
    merged_df = pd.merge(complete_dates_df, df_filtered, how='left', on='date')
    missing_dates = set()

    for idx in range(1, len(merged_df) - 1):
        if pd.isna(merged_df.iloc[idx][measurement_columns]).any():
            prev_row = merged_df.iloc[idx - 1]
            next_row = merged_df.iloc[idx + 1]
            for col in measurement_columns:
                if pd.isna(merged_df.at[idx, col]):
                    avg_value = (prev_row[col] + next_row[col]) / 2
                    merged_df.at[idx, col] = avg_value
                    missing_dates.add(merged_df.at[idx, 'date'].strftime('%Y-%m-%d %H:%M:%S'))

    return merged_df[['date'] + measurement_columns], list(missing_dates)

def detect_and_correct_failures(df, measurement_columns, negative_variables):
    df['date'] = pd.to_datetime(df['date'])
    df['time_diff'] = df['date'].diff().dt.total_seconds()
    mode_freq = df['time_diff'].mode()[0]
    freq_mask = df['time_diff'] == mode_freq

    # Calcular variaciones promedio absolutas para cada variable
    avg_variations = {
        col: abs(df[col].diff())[freq_mask].mean()
        for col in measurement_columns
    }

    # Calcular correlaciones entre variables
    correlations = df[measurement_columns].corr()
    
    detected_failures = []

    for i in range(1, len(df)):
        # Verificar si el intervalo de tiempo es el correcto (moda)
        if df['time_diff'].iloc[i] != mode_freq:
            continue

        for col in measurement_columns:
            val = df.at[i, col]
            prev_val = df.at[i - 1, col]
            
            if pd.isna(val) or pd.isna(prev_val):
                continue

            avg_var = avg_variations.get(col, 0)
            if avg_var == 0:
                continue

            # Falla Tipo 1: Variación anormal
            if abs(val - prev_val) >= 10 * avg_var:
                # Buscar las 2 variables de mayor correlación
                corr_vars = correlations[col][(correlations[col].abs() > 0.7) & (correlations[col].abs() < 1.0)]
                corr_vars = corr_vars.sort_values(ascending=False).head(2)
                
                is_failure = True
                if len(corr_vars) >= 2:
                    # Verificar si las variables correlacionadas también varían significativamente
                    corr_vars_changed = 0
                    for corr_col in corr_vars.index:
                        corr_val = df.at[i, corr_col]
                        corr_prev_val = df.at[i - 1, corr_col]
                        if not pd.isna(corr_val) and not pd.isna(corr_prev_val):
                            corr_avg_var = avg_variations.get(corr_col, 0)
                            if abs(corr_val - corr_prev_val) >= 3 * corr_avg_var:
                                corr_vars_changed += 1
                    
                    if corr_vars_changed >= 2:
                        is_failure = False

                if is_failure:
                    # Calcular valor esperado
                    # Encontrar la variable de mayor correlación
                    best_corr = correlations[col][(correlations[col].abs() > 0.7) & (correlations[col].abs() < 1.0)]
                    if not best_corr.empty:
                        best_corr_col = best_corr.index[0]
                        best_corr_val = df.at[i, best_corr_col]
                        best_corr_prev_val = df.at[i - 1, best_corr_col]
                        
                        if not pd.isna(best_corr_val) and not pd.isna(best_corr_prev_val):
                            # Determinar si sumar o restar basado en la variación de la variable correlacionada
                            if best_corr_val > best_corr_prev_val:
                                expected_val = prev_val + avg_var
                            else:
                                expected_val = prev_val - avg_var
                            
                            # Si el valor esperado es negativo y la variable no lo permite, usar 0
                            if expected_val < 0 and col not in negative_variables:
                                expected_val = 0
                            
                            df.at[i, col] = expected_val
                            detected_failures.append({
                                'date': df.at[i, 'date'],
                                'variable': col,
                                'original_value': val,
                                'expected_value': expected_val,
                                'error_type': 'variation'
                            })

            # Falla Tipo 2: Valor negativo no permitido
            elif val < 0 and col not in negative_variables:
                expected_val = 0
                df.at[i, col] = expected_val
                detected_failures.append({
                    'date': df.at[i, 'date'],
                    'variable': col,
                    'original_value': val,
                    'expected_value': expected_val,
                    'error_type': 'negative'
                })

    df.drop(columns=['time_diff'], inplace=True)
    return df, detected_failures

@app.route('/download/<filename>')
@login_required
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 10000)))

