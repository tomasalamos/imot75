import os
import warnings
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, session, flash, send_file
import pandas as pd
import numpy as np
from datetime import datetime
from functools import wraps
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.utils import secure_filename
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from fpdf import FPDF
import json
import matplotlib
matplotlib.use('Agg')  # Configurar el backend no interactivo
import matplotlib.pyplot as plt
import seaborn as sns
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

warnings.filterwarnings('ignore')

app = Flask(__name__, 
    static_url_path='',
    static_folder='static')
app.secret_key = 'supersecretkey123'  # Change to a secure key

# Configuración de la base de datos
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Modelo de Usuario
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

UPLOAD_FOLDER = 'upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'auth'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Crear la base de datos
with app.app_context():
    db.create_all()

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

@app.route('/auth', methods=['GET', 'POST'])
def auth():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        action = request.form['action']

        if action == 'login':
            user = User.query.filter_by(username=username).first()
            if user and user.check_password(password):
                login_user(user)
                return redirect(url_for('upload'))
            else:
                flash('Usuario o contraseña incorrectos', 'error')

        elif action == 'register':
            if User.query.filter_by(username=username).first():
                flash('El usuario ya existe', 'error')
            else:
                email = request.form.get('email', '')
                if not email:
                    flash('El email es requerido', 'error')
                    return redirect(url_for('auth'))
                
                if User.query.filter_by(email=email).first():
                    flash('El email ya está registrado', 'error')
                    return redirect(url_for('auth'))

                user = User(username=username, email=email)
                user.set_password(password)
                db.session.add(user)
                db.session.commit()
                flash('Usuario registrado exitosamente. Ahora puedes iniciar sesión.', 'success')

    return render_template('auth.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Has cerrado sesión', 'info')
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

def calculate_negative_percentages(df):
    negative_vars = {}
    for column in df.select_dtypes(include=['float64', 'int64']).columns:
        negative_count = (df[column] < 0).sum()
        total_count = df[column].count()
        if total_count > 0:
            percentage = (negative_count / total_count) * 100
            if percentage > 0:
                negative_vars[column] = percentage
    return negative_vars

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
        
        # Calcular variables con valores negativos
        negative_vars = calculate_negative_percentages(df)

        return render_template('form.html', 
                             variables=numeric_columns, 
                             negative_vars=negative_vars,
                             min_date=min_date, 
                             max_date=max_date)

    except Exception as e:
        flash(f'Error processing file: {str(e)}', 'error')
        return redirect(url_for('upload'))

@app.route('/process', methods=['POST'])
@login_required
def process():
    if 'uploaded_file' not in session:
        flash('Please upload a file first', 'error')
        return redirect(url_for('upload'))
        
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], session['uploaded_file'])
        df = pd.read_csv(filepath)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])

        selected_vars = request.form.getlist('variables')
        negative_vars = request.form.getlist('negative_variables')

        # Guardar datos en la sesión
        session['selected_vars'] = selected_vars
        session['negative_vars'] = negative_vars
        session['start_date'] = df['date'].min().strftime('%Y-%m-%d %H:%M:%S')
        session['end_date'] = df['date'].max().strftime('%Y-%m-%d %H:%M:%S')

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

        return redirect(url_for('results'))
                           
    except Exception as e:
        flash(f'Error processing file: {str(e)}', 'error')
        return redirect(url_for('form'))

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

@app.route('/additionals')
def additionals():
    if 'selected_vars' not in session:
        return redirect(url_for('upload'))
    
    # Obtener las variables seleccionadas de la sesión
    selected_vars = session.get('selected_vars', [])
    
    return render_template('additionals.html', variables=selected_vars)

@app.route('/results')
@login_required
def results():
    if 'uploaded_file' not in session:
        flash('Please upload a file first', 'error')
        return redirect(url_for('upload'))
        
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], session['uploaded_file'])
        df = pd.read_csv(filepath)
        
        return render_template('results.html',
                           selected_vars=session.get('selected_vars', []),
                           negative_vars=session.get('negative_vars', []),
                           start=session.get('start_date', ''),
                           end=session.get('end_date', ''))
    except Exception as e:
        flash(f'Error processing file: {str(e)}', 'error')
        return redirect(url_for('upload'))

@app.route('/download_pdf')
def download_pdf():
    if 'selected_vars' not in session:
        return redirect(url_for('upload'))
    
    # Obtener las variables seleccionadas de la sesión
    selected_vars = session.get('selected_vars', [])
    
    # Generar el PDF solo con las variables seleccionadas
    pdf_path = generate_pdf_report(selected_vars)
    return send_file(pdf_path, as_attachment=True, download_name='analysis_report.pdf')

def generate_pdf_report(selected_vars):
    if 'uploaded_file' not in session:
        raise Exception('No file uploaded')
        
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], session['uploaded_file'])
    df = pd.read_csv(filepath)
    
    # Crear el PDF
    pdf = FPDF()
    pdf.add_page()
    
    # Título
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Data Analysis Report', 0, 1, 'C')
    pdf.ln(10)
    
    # Fechas
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Date Range:', 0, 1)
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 10, f'From: {df["date"].min()}', 0, 1)
    pdf.cell(0, 10, f'To: {df["date"].max()}', 0, 1)
    pdf.ln(10)
    
    # Variables y estadísticas (solo las seleccionadas)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Variable Statistics:', 0, 1)
    
    for column in selected_vars:
        if column in df.columns:
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, f'\n{column}:', 0, 1)
            pdf.set_font('Arial', '', 12)
            
            stats = df[column].describe()
            pdf.cell(0, 10, f'Mean: {stats["mean"]:.2f}', 0, 1)
            pdf.cell(0, 10, f'Median: {stats["50%"]:.2f}', 0, 1)
            pdf.cell(0, 10, f'Std Dev: {stats["std"]:.2f}', 0, 1)
            pdf.cell(0, 10, f'Min: {stats["min"]:.2f}', 0, 1)
            pdf.cell(0, 10, f'Max: {stats["max"]:.2f}', 0, 1)
    
    # Fechas faltantes
    pdf.ln(10)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Missing Dates:', 0, 1)
    pdf.set_font('Arial', '', 12)
    
    date_range = pd.date_range(start=df['date'].min(), end=df['date'].max(), freq='D')
    missing_dates = date_range.difference(df['date'])
    if len(missing_dates) > 0:
        for date in missing_dates:
            pdf.cell(0, 10, str(date.date()), 0, 1)
    else:
        pdf.cell(0, 10, 'No missing dates found', 0, 1)
    
    # Fallas detectadas (detalle completo)
    pdf.ln(10)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Detected Failures:', 0, 1)
    pdf.set_font('Arial', '', 12)
    
    # Leer el archivo de fallas detectadas
    failures_path = os.path.join(app.config['UPLOAD_FOLDER'], 'detected_failures.csv')
    if os.path.exists(failures_path):
        failures_df = pd.read_csv(failures_path)
        if not failures_df.empty:
            # Agrupar fallas por variable
            for var in selected_vars:
                var_failures = failures_df[failures_df['variable'] == var]
                if not var_failures.empty:
                    pdf.set_font('Arial', 'B', 12)
                    pdf.cell(0, 10, f'\n{var}:', 0, 1)
                    pdf.set_font('Arial', '', 12)
                    
                    for _, failure in var_failures.iterrows():
                        pdf.cell(0, 10, f'Date: {failure["date"]}', 0, 1)
                        pdf.cell(0, 10, f'Original Value: {failure["original_value"]:.2f}', 0, 1)
                        pdf.cell(0, 10, f'Expected Value: {failure["expected_value"]:.2f}', 0, 1)
                        pdf.cell(0, 10, f'Error Type: {failure["error_type"]}', 0, 1)
                        pdf.ln(5)
        else:
            pdf.cell(0, 10, 'No failures detected', 0, 1)
    else:
        pdf.cell(0, 10, 'No failures detected', 0, 1)
    
    # Guardar el PDF
    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], 'analysis_report.pdf')
    pdf.output(pdf_path)
    
    return pdf_path

@app.route('/generate_graphs', methods=['POST'])
def generate_graphs():
    if 'uploaded_file' not in session or 'selected_vars' not in session:
        return redirect(url_for('upload'))
    
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], session['uploaded_file'])
        df = pd.read_csv(filepath)
        df['date'] = pd.to_datetime(df['date'])
        
        # Obtener las variables seleccionadas de la sesión
        selected_vars = session.get('selected_vars', [])
        
        # Obtener el tipo de gráfico y las variables seleccionadas del formulario
        graph_type = request.form.get('graph_type')
        selected_variables = request.form.getlist('variables[]')
        
        # Verificar que las variables seleccionadas estén en la lista de variables permitidas
        selected_variables = [var for var in selected_variables if var in selected_vars]
        
        if not selected_variables:
            flash('Please select at least one variable', 'error')
            return redirect(url_for('additionals'))
        
        # Limpiar cualquier figura existente
        plt.close('all')
        
        # Generar el gráfico según el tipo seleccionado
        fig = plt.figure(figsize=(12, 6))
        
        if graph_type == 'density':
            if len(selected_variables) != 1:
                flash('Density plot requires exactly one variable', 'error')
                return redirect(url_for('additionals'))
            sns.kdeplot(data=df[selected_variables[0]])
            plt.title(f'Density Plot - {selected_variables[0]}')
            
        elif graph_type == 'correlation':
            if len(selected_variables) < 2:
                flash('Correlation matrix requires at least two variables', 'error')
                return redirect(url_for('additionals'))
            corr_matrix = df[selected_variables].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
            plt.title('Correlation Matrix')
            
        elif graph_type == 'simple':
            if len(selected_variables) != 1:
                flash('Simple line plot requires exactly one variable', 'error')
                return redirect(url_for('additionals'))
            plt.plot(df['date'], df[selected_variables[0]])
            plt.title(f'Simple Line Plot - {selected_variables[0]}')
            plt.xticks(rotation=45)
            
        elif graph_type == 'dual_axis':
            if len(selected_variables) != 2:
                flash('Dual axis plot requires exactly two variables', 'error')
                return redirect(url_for('additionals'))
            ax1 = plt.gca()
            ax2 = ax1.twinx()
            ax1.plot(df['date'], df[selected_variables[0]], 'b-')
            ax2.plot(df['date'], df[selected_variables[1]], 'r-')
            ax1.set_xlabel('Date')
            ax1.set_ylabel(selected_variables[0], color='b')
            ax2.set_ylabel(selected_variables[1], color='r')
            plt.title('Dual Axis Plot')
            plt.xticks(rotation=45)
            
        elif graph_type == 'box':
            if len(selected_variables) != 1:
                flash('Box plot requires exactly one variable', 'error')
                return redirect(url_for('additionals'))
            sns.boxplot(data=df[selected_variables[0]])
            plt.title(f'Box Plot - {selected_variables[0]}')
            
        elif graph_type == 'scatter':
            if len(selected_variables) != 2:
                flash('Scatter plot requires exactly two variables', 'error')
                return redirect(url_for('additionals'))
            plt.scatter(df[selected_variables[0]], df[selected_variables[1]])
            plt.xlabel(selected_variables[0])
            plt.ylabel(selected_variables[1])
            plt.title('Scatter Plot')
            
        elif graph_type == 'stacked_bar':
            if len(selected_variables) < 2:
                flash('Stacked bar chart requires at least two variables', 'error')
                return redirect(url_for('additionals'))
            df[selected_variables].plot(kind='bar', stacked=True)
            plt.title('Stacked Bar Chart')
            plt.xticks(rotation=45)
            
        elif graph_type == 'multi_line':
            if len(selected_variables) < 2:
                flash('Multiple line plot requires at least two variables', 'error')
                return redirect(url_for('additionals'))
            for var in selected_variables:
                plt.plot(df['date'], df[var], label=var)
            plt.title('Multiple Line Plot')
            plt.legend()
            plt.xticks(rotation=45)
            
        else:
            flash('Invalid graph type selected', 'error')
            return redirect(url_for('additionals'))
        
        # Guardar el gráfico en la carpeta static
        static_folder = os.path.join(app.root_path, 'static')
        graph_path = os.path.join(static_folder, 'generated_graph.png')
        plt.tight_layout()
        plt.savefig(graph_path)
        
        # Limpiar la figura actual
        plt.close(fig)
        
        return render_template('graph.html')
        
    except Exception as e:
        # Asegurarse de limpiar cualquier figura en caso de error
        plt.close('all')
        flash(f'Error generating graph: {str(e)}', 'error')
        return redirect(url_for('additionals'))

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)