from flask import Flask, request, jsonify, send_from_directory, json
from flask_cors import CORS
import sqlite3
from sqlite3 import Error as SQLiteError
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import os
from datetime import datetime, timedelta
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
import time
import numpy as np
import uuid
from scipy import stats

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'db', 'xlsx', 'xls', 'csv'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

GEMINI_API_KEY = 'AIzaSyDYCz04SFQjLueHGfPdpy_UD7m22U-eTko'
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

# Configure retry strategy and HTTP session
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
)
http = requests.Session()
http.mount("https://", HTTPAdapter(max_retries=retry_strategy))
http.mount("http://", HTTPAdapter(max_retries=retry_strategy))

REPORTS_STORAGE = {}

class DBConnection:
    _instance = None
    _db_initialized = False

    @classmethod
    def get_instance(cls):
        return cls._instance

    @classmethod
    def set_instance(cls, connection_params=None):
        if cls._instance:
            try:
                cls._instance.close()
            except:
                pass
        cls._instance = connection_params
        cls._db_initialized = bool(connection_params)

    @classmethod
    def is_initialized(cls):
        return cls._db_initialized

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    """Serve static files including report.html"""
    return send_from_directory('.', filename)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def ensure_upload_dir():
    """Ensure upload directory exists and is accessible"""
    try:
        # Get absolute path for uploads directory relative to app.py
        current_dir = os.path.dirname(os.path.abspath(__file__))
        upload_dir = os.path.join(current_dir, UPLOAD_FOLDER)
        
        # Create directory if it doesn't exist
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)
            
        # Verify directory is writable
        test_file = os.path.join(upload_dir, 'test.txt')
        try:
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
        except Exception as e:
            print(f"Directory not writable: {e}")
            return None
            
        return upload_dir
    except Exception as e:
        print(f"Error creating upload directory: {e}")
        return None

def cleanup_previous_files():
    """Clean up any existing database and upload files"""
    try:
        # Clean up database file
        if os.path.exists('no.db'):
            try:
                DBConnection.set_instance(None)
                os.remove('no.db')
            except Exception as e:
                print(f"Warning: Could not remove database: {e}")

        # Clean up uploads directory
        upload_dir = ensure_upload_dir()
        if upload_dir:
            for filename in os.listdir(upload_dir):
                try:
                    file_path = os.path.join(upload_dir, filename)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                except Exception as e:
                    print(f"Warning: Could not remove file {filename}: {e}")
    except Exception as e:
        print(f"Warning: Cleanup error: {e}")

def process_excel_file(file_path):
    try:
        # Ensure file exists and is readable
        if not os.path.isfile(file_path):
            raise Exception(f"File not found: {file_path}")
            
        try:
            with open(file_path, 'rb') as f:
                # Just test if file is readable
                f.read(1)
        except Exception as e:
            raise Exception(f"File not readable: {str(e)}")

        # Process file based on type
        if file_path.endswith('.csv'):
            for encoding in ['utf-8', 'latin1', 'cp1252']:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    print(f"Successfully read CSV with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    raise Exception(f"Error reading CSV: {str(e)}")
            else:
                raise Exception("Could not read CSV with any encoding")
        else:
            df = pd.read_excel(file_path)

        # Clean column names
        df.columns = df.columns.astype(str).str.strip()
        df.columns = df.columns.str.replace('[^a-zA-Z0-9_]', '_', regex=True)
        df.columns = df.columns.str.lower()
        
        # Replace empty values with None
        df = df.replace(['', 'nan', 'NaN', 'NULL'], None)
        df = df.where(pd.notnull(df), None)
        
        try:
            conn = sqlite3.connect('no.db')
            # Convert DataFrame to SQL
            df.to_sql('no', conn, if_exists='replace', index=False)
            conn.close()
            return True
        except sqlite3.Error as e:
            print(f"SQLite error: {str(e)}")
            raise Exception(f"Database error: {str(e)}")
            
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        raise Exception(f"File processing error: {str(e)}")

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        # Ensure upload directory exists
        upload_dir = ensure_upload_dir()
        if not upload_dir:
            return jsonify({'error': 'Could not create or access upload directory'})

        if 'file' not in request.files:
            return jsonify({'error': 'No file selected'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'})
        
        if file and allowed_file(file.filename):
            try:
                # Create a secure filename
                filename = os.path.join(upload_dir, os.path.basename(file.filename))
                
                # Save uploaded file
                file.save(filename)
                print(f"File saved to: {filename}")

                try:
                    if filename.endswith('.db'):
                        db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'no.db')
                        os.replace(filename, db_path)
                        DBConnection._db_initialized = True
                    else:
                        process_excel_file(filename)
                        DBConnection._db_initialized = True
                    return jsonify({'message': 'File uploaded successfully'})
                except Exception as e:
                    return jsonify({'error': f"Processing error: {str(e)}"})
                finally:
                    # Clean up uploaded file
                    try:
                        if os.path.exists(filename):
                            os.remove(filename)
                    except Exception as e:
                        print(f"Warning: Could not remove temporary file: {e}")
            except Exception as e:
                return jsonify({'error': f'Error saving file: {str(e)}'})
        
        return jsonify({'error': 'Invalid file type'})
        
    except Exception as e:
        DBConnection._db_initialized = False
        return jsonify({'error': str(e)})

@app.route('/connect-db', methods=['POST'])
def connect_db():
    try:
        data = request.json
        conn = psycopg2.connect(
            host=data['host'],
            port=data['port'],
            database=data['database'],
            user=data['user'],
            password=data['password']
        )
        # Test connection
        with conn.cursor() as cursor:
            cursor.execute('SELECT 1')
        
        # Store connection parameters
        DBConnection.set_instance(data)
        
        return jsonify({'message': 'Connected successfully'})
    except Exception as e:
        return jsonify({'error': str(e)})

def get_table_schema():
    if not DBConnection.is_initialized():
        return {}
        
    try:
        # Check for direct database connection first
        conn_params = DBConnection.get_instance()
        if conn_params:
            try:
                conn = psycopg2.connect(**conn_params)
                cursor = conn.cursor()
                
                # Get all tables
                cursor.execute("""
                    SELECT table_name, column_name 
                    FROM information_schema.columns 
                    WHERE table_schema = 'public'
                    ORDER BY table_name, ordinal_position;
                """)
                
                schema_info = {}
                for table_name, column_name in cursor.fetchall():
                    if table_name not in schema_info:
                        schema_info[table_name] = []
                    schema_info[table_name].append(column_name)
                
                conn.close()
                return schema_info
            except Exception:
                pass
        
        # Fallback to SQLite file
        if not os.path.exists('no.db'):
            return {}
            
        conn = sqlite3.connect('no.db')
        if not conn:
            return {}
            
        cursor = conn.cursor()
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        schema_info = {}
        for table in tables:
            table_name = table[0]
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            schema_info[table_name] = [col[1] for col in columns]
            
        conn.close()
        return schema_info
    except Exception as e:
        print(f"Schema error: {str(e)}")
        return {}

def generate_sql_with_gemini(query):
    headers = {
        'Content-Type': 'application/json'
    }
    
    # Get current database schema
    schema_info = get_table_schema()
    schema_text = "Available tables and their columns:\n"
    for table, columns in schema_info.items():
        schema_text += f"Table '{table}': {', '.join(columns)}\n"
    
    data = {
        "contents": [{
            "parts": [{
                "text": f"""You are a SQL expert. Convert the following natural language query to SQL.
                        Database Schema:
                        {schema_text}
                        
                        Rules:
                        1. Only return the SQL query, no explanations
                        2. Use only the tables and columns from the schema
                        3. If table doesn't exist, return 'ERROR: Required table not found'
                        
                        Natural Language Query: {query}"""
            }]
        }]
    }
    
    try:
        response = http.post(
            f"{GEMINI_URL}?key={GEMINI_API_KEY}",
            headers=headers,
            json=data,
            timeout=10,
            verify=True
        )
        
        if response.status_code == 200:
            result = response.json()
            sql = result['candidates'][0]['content']['parts'][0]['text'].strip()
            if sql.startswith('ERROR:'):
                raise Exception(sql)
            return sql.replace('```sql', '').replace('```', '').strip()
        else:
            raise Exception(f"API Error: {response.status_code} - {response.text}")
    except Exception as e:
        raise Exception(f"Unexpected error: {str(e)}")

@app.route('/schema', methods=['GET'])
def get_schema():
    return jsonify(get_table_schema())

# Initialize SQLite database
def init_db():
    try:
        cleanup_previous_files()
        
        # Create new database
        conn = sqlite3.connect('no.db')
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS temp_table
            (id INTEGER PRIMARY KEY)
        ''')
        conn.commit()
        conn.close()
        DBConnection._db_initialized = False
        return True
    except SQLiteError as e:
        print(f"Database initialization error: {str(e)}")
        DBConnection._db_initialized = False
        return False

def analyze_data_for_visualization(data):
    """Analyze data and suggest appropriate visualizations"""
    if not data or len(data) == 0:
        return {"suitable_charts": [], "reason": "No data available"}

    try:
        df = pd.DataFrame(data)
        columns = df.columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        try:
            date_cols = pd.to_datetime(df[columns], errors='coerce').notna().all()
            date_cols = columns[date_cols]
        except:
            date_cols = []
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        num_rows = len(df)
        num_cols = len(columns)
        
        suitable_charts = []
        analysis = {
            "row_count": num_rows,
            "column_count": num_cols,
            "numeric_columns": list(numeric_cols),
            "date_columns": list(date_cols),
            "categorical_columns": list(categorical_cols)
        }

        # Bar Chart
        if len(numeric_cols) > 0 and num_rows <= 50:
            suitable_charts.append({
                "type": "bar",
                "confidence": 0.9 if num_rows <= 20 else 0.7,
                "suggested_columns": list(numeric_cols)[:3]
            })

        # Line Chart
        if len(date_cols) > 0 and len(numeric_cols) > 0:
            suitable_charts.append({
                "type": "line",
                "confidence": 0.9,
                "suggested_columns": {
                    "x": list(date_cols)[0],
                    "y": list(numeric_cols)[0]
                }
            })

        # Pie Chart
        if len(categorical_cols) > 0 and len(numeric_cols) > 0 and num_rows <= 10:
            suitable_charts.append({
                "type": "pie",
                "confidence": 0.8,
                "suggested_columns": {
                    "labels": list(categorical_cols)[0],
                    "values": list(numeric_cols)[0]
                }
            })

        # Scatter Plot
        if len(numeric_cols) >= 2:
            correlations = []
            for i in range(len(numeric_cols)):
                for j in range(i+1, len(numeric_cols)):
                    corr = df[numeric_cols[i]].corr(df[numeric_cols[j]])
                    if not np.isnan(corr):
                        correlations.append((numeric_cols[i], numeric_cols[j], abs(corr)))
            
            if correlations:
                best_correlation = max(correlations, key=lambda x: x[2])
                suitable_charts.append({
                    "type": "scatter",
                    "confidence": min(best_correlation[2], 0.9),
                    "suggested_columns": {
                        "x": best_correlation[0],
                        "y": best_correlation[1]
                    }
                })

        # Heatmap
        if len(numeric_cols) > 2:
            suitable_charts.append({
                "type": "heatmap",
                "confidence": 0.7,
                "suggested_columns": list(numeric_cols)
            })

        return {
            "suitable_charts": sorted(suitable_charts, key=lambda x: x["confidence"], reverse=True),
            "analysis": analysis
        }

    except Exception as e:
        return {"error": str(e)}

@app.route('/generate-query', methods=['POST'])
def generate_query():
    try:
        nl_query = request.json['query']
        sql_query = generate_sql_with_gemini(nl_query)
        
        conn_params = DBConnection.get_instance()
        if conn_params:
            conn = psycopg2.connect(**conn_params)
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute(sql_query)
            results = cursor.fetchall()
            conn.close()
        else:
            conn = sqlite3.connect('no.db')
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(sql_query)
            results = [dict(row) for row in cursor.fetchall()]
            conn.close()

        # Analyze results for visualization
        visualization_suggestions = analyze_data_for_visualization(results)
        
        return jsonify({
            'sql_query': sql_query,
            'results': results,
            'visualizations': visualization_suggestions
        })
        
    except Exception as e:
        return jsonify({
            'sql_query': '',
            'error': str(e)
        })

def convert_to_native_types(obj):
    """Convert NumPy/Pandas types to native Python types"""
    if isinstance(obj, dict):
        return {k: convert_to_native_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(i) for i in obj]
    elif isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj) if not np.isnan(obj) else None
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, pd.Series):
        return [convert_to_native_types(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return [convert_to_native_types(v) for v in obj]
    elif pd.isna(obj):  # Handle pandas NA/NaN values
        return None
    return obj

@app.route('/report/<report_id>')
def get_report(report_id):
    try:
        report = REPORTS_STORAGE.get(report_id)
        if not report:
            print(f"Report {report_id} not found in storage. Available reports: {list(REPORTS_STORAGE.keys())}")
            return jsonify({"error": "Report not found"}), 404
            
        # Ensure the report has all required fields
        required_fields = ['overview', 'statistics', 'quality']
        if not all(field in report for field in required_fields):
            return jsonify({"error": "Invalid report structure"}), 500
            
        return jsonify(convert_to_native_types(report))
    except Exception as e:
        print(f"Error retrieving report: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/generate-report', methods=['POST'])
def create_report():
    try:
        data = request.json.get('data')
        if not data or not isinstance(data, list) or len(data) == 0:
            print("Invalid data format received:", data)
            return jsonify({"error": "Invalid or empty data provided"}), 400

        # Validate data structure
        for row in data:
            if not isinstance(row, dict):
                print("Invalid row format:", row)
                return jsonify({"error": "Invalid data format"}), 400

        print(f"Generating report for {len(data)} rows of data")
        report = generate_data_report(data)
        
        if not report:
            print("Failed to generate report")
            return jsonify({"error": "Failed to generate report"}), 500

        print(f"Successfully generated report with ID: {report['id']}")
        return jsonify({
            "report_id": report["id"],
            "message": "Report generated successfully"
        })

    except Exception as e:
        print(f"Error in create_report: {str(e)}")
        return jsonify({"error": f"Report generation failed: {str(e)}"}), 500

def detect_data_context(df):
    """Detect the type of data based on column names and values"""
    columns = set(df.columns.str.lower())
    
    # Education/School context indicators
    education_indicators = {
        'student', 'grade', 'attendance', 'class', 'subject', 'teacher', 'school',
        'score', 'semester', 'exam', 'marks', 'admission', 'course'
    }
    
    # Business/Sales context indicators
    business_indicators = {
        'sales', 'revenue', 'customer', 'product', 'price', 'order', 'inventory',
        'employee', 'profit', 'cost', 'invoice', 'payment', 'transaction'
    }
    
    # Healthcare context indicators
    healthcare_indicators = {
        'patient', 'diagnosis', 'treatment', 'doctor', 'hospital', 'medical',
        'prescription', 'symptoms', 'admission', 'discharge', 'medication'
    }

    # Count matches for each context
    edu_matches = len(columns.intersection(education_indicators))
    business_matches = len(columns.intersection(business_indicators))
    health_matches = len(columns.intersection(healthcare_indicators))
    
    # Determine primary context
    if max(edu_matches, business_matches, health_matches) == 0:
        return "general"
    
    contexts = []
    if edu_matches > 1:
        contexts.append(("education", edu_matches))
    if business_matches > 1:
        contexts.append(("business", business_matches))
    if health_matches > 1:
        contexts.append(("healthcare", health_matches))
    
    if not contexts:
        return "general"
        
    # Return the context with highest matches
    primary_context = max(contexts, key=lambda x: x[1])[0]
    return primary_context

def generate_context_insights(df, context, statistics):
    """Generate domain-specific insights"""
    insights = []
    
    if context == "education":
        if 'attendance' in df.columns:
            avg_attendance = df['attendance'].mean()
            insights.append({
                "type": "attendance",
                "severity": "medium" if avg_attendance < 85 else "low",
                "message": f"Average attendance rate is {avg_attendance:.1f}%"
            })
        
        grade_columns = [col for col in df.columns if 'grade' in col.lower() or 'score' in col.lower()]
        if grade_columns:
            for col in grade_columns:
                avg_score = df[col].mean()
                insights.append({
                    "type": "academic",
                    "severity": "high" if avg_score < 60 else "medium" if avg_score < 75 else "low",
                    "message": f"Average {col}: {avg_score:.1f}"
                })
    
    elif context == "business":
        revenue_cols = [col for col in df.columns if 'revenue' in col.lower() or 'sales' in col.lower()]
        if revenue_cols:
            for col in revenue_cols:
                total_revenue = df[col].sum()
                trend = df[col].pct_change().mean() * 100
                insights.append({
                    "type": "revenue",
                    "severity": "medium",
                    "message": f"Total {col}: ${total_revenue:,.2f}, Trend: {trend:+.1f}%"
                })
        
        if 'profit' in df.columns:
            profit_margin = (df['profit'] / df['revenue']).mean() * 100
            insights.append({
                "type": "profitability",
                "severity": "high" if profit_margin < 10 else "medium" if profit_margin < 20 else "low",
                "message": f"Average profit margin: {profit_margin:.1f}%"
            })
    
    elif context == "healthcare":
        if 'patient_count' in df.columns:
            avg_patients = df['patient_count'].mean()
            insights.append({
                "type": "patient_load",
                "severity": "medium",
                "message": f"Average patient count: {avg_patients:.0f} per period"
            })

    return insights

def analyze_trends(df, context):
    """Analyze trends based on context"""
    trends = {}
    
    try:
        if context == "business":
            # Sales trends
            if 'sales' in df.columns:
                trends['sales'] = {
                    'current': float(df['sales'].tail(30).sum()),
                    'previous': float(df['sales'].tail(60).head(30).sum()),
                    'growth': float(df['sales'].pct_change().mean() * 100)
                }
            
            # Product performance
            if 'product' in df.columns and 'sales' in df.columns:
                product_performance = df.groupby('product')['sales'].agg([
                    ('total_sales', 'sum'),
                    ('avg_sales', 'mean')
                ]).sort_values('total_sales', ascending=False)
                
                trends['products'] = {
                    'top': product_performance.head(5).to_dict('index'),
                    'bottom': product_performance.tail(5).to_dict('index')
                }

        elif context == "education":
            # Academic performance trends
            grade_cols = [col for col in df.columns if 'grade' in col.lower() or 'score' in col.lower()]
            if grade_cols:
                trends['academic'] = {
                    'average_scores': {col: float(df[col].mean()) for col in grade_cols},
                    'improvement': {col: float(df[col].diff().mean()) for col in grade_cols}
                }

        elif context == "healthcare":
            # Patient care metrics
            if 'patient_count' in df.columns:
                trends['patients'] = {
                    'daily_average': float(df['patient_count'].mean()),
                    'peak_days': df.nlargest(5, 'patient_count')[['date', 'patient_count']].to_dict('records')
                }

    except Exception as e:
        print(f"Error analyzing trends: {str(e)}")
        trends['error'] = str(e)

    return trends

def generate_context_recommendations(context, metrics, trends):
    """Generate context-specific recommendations"""
    recommendations = []
    
    if context == "business":
        if trends.get('sales', {}).get('growth', 0) < 0:
            recommendations.append({
                "title": "Sales Growth Strategy",
                "description": "Sales are declining. Consider these improvements:",
                "action": "• Review pricing strategy\n• Launch promotional campaign\n• Analyze competitor pricing\n• Focus on top-performing products"
            })
        
        # Add product-specific recommendations
        if trends.get('products'):
            top_products = list(trends['products']['top'].keys())[:3]
            recommendations.append({
                "title": "Product Strategy",
                "description": f"Focus on top performers: {', '.join(top_products)}",
                "action": "• Increase inventory for top products\n• Bundle top products with lower performers\n• Analyze what makes top products successful"
            })

    # Add similar blocks for education and healthcare contexts
    
    return recommendations

def generate_business_metrics(df):
    """Generate comprehensive business metrics and KPIs"""
    metrics = {}
    try:
        if 'sales' in df.columns:
            # Sales Analysis
            metrics['sales'] = {
                'total_revenue': float(df['sales'].sum()),
                'average_transaction': float(df['sales'].mean()),
                'peak_sales': float(df['sales'].max()),
                'growth_rate': float(df['sales'].pct_change().mean() * 100),
                'monthly_trend': df.groupby(pd.Grouper(freq='M'))['sales'].sum().to_dict()
            }
            
        if 'product' in df.columns:
            # Product Analysis
            product_metrics = df.groupby('product').agg({
                'sales': ['sum', 'mean', 'count'],
                'profit': ['sum', 'mean'] if 'profit' in df.columns else None
            }).dropna(axis=1, how='all')
            
            metrics['products'] = {
                'top_performers': product_metrics.nlargest(5, ('sales', 'sum')).to_dict(),
                'underperforming': product_metrics.nsmallest(5, ('sales', 'sum')).to_dict(),
                'product_count': len(df['product'].unique())
            }
            
        if 'customer' in df.columns:
            # Customer Analysis
            metrics['customer'] = {
                'total_customers': len(df['customer'].unique()),
                'avg_customer_value': float(df.groupby('customer')['sales'].sum().mean()),
                'retention_rate': calculate_retention_rate(df),
                'segment_analysis': perform_customer_segmentation(df)
            }
            
        if 'date' in df.columns:
            # Seasonal Analysis
            df['month'] = pd.to_datetime(df['date']).dt.month
            seasonal_pattern = df.groupby('month')['sales'].mean()
            metrics['seasonality'] = {
                'peak_month': int(seasonal_pattern.idxmax()),
                'low_month': int(seasonal_pattern.idxmin()),
                'seasonal_variance': float(seasonal_pattern.std())
            }
            
    except Exception as e:
        print(f"Error in business metrics calculation: {str(e)}")
    return metrics

def calculate_retention_rate(df):
    """Calculate customer retention rate"""
    try:
        if 'date' in df.columns and 'customer' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            periods = df.groupby(pd.Grouper(key='date', freq='M'))
            
            retention_rates = []
            previous_customers = set()
            
            for _, period in periods:
                current_customers = set(period['customer'].unique())
                if previous_customers:
                    retention_rate = len(current_customers & previous_customers) / len(previous_customers)
                    retention_rates.append(retention_rate)
                previous_customers = current_customers
                
            return float(np.mean(retention_rates) * 100) if retention_rates else None
    except Exception:
        return None

def perform_customer_segmentation(df):
    """Perform RFM (Recency, Frequency, Monetary) analysis"""
    try:
        if all(col in df.columns for col in ['customer', 'date', 'sales']):
            df['date'] = pd.to_datetime(df['date'])
            last_date = df['date'].max()
            
            rfm = df.groupby('customer').agg({
                'date': lambda x: (last_date - x.max()).days,  # Recency
                'sales': ['count', 'sum']  # Frequency and Monetary
            })
            
            # Segment customers
            r_labels = range(4, 0, -1)
            f_labels = range(1, 5)
            m_labels = range(1, 5)
            
            r_quartiles = pd.qcut(rfm['date'], q=4, labels=r_labels)
            f_quartiles = pd.qcut(rfm['sales']['count'], q=4, labels=f_labels)
            m_quartiles = pd.qcut(rfm['sales']['sum'], q=4, labels=m_labels)
            
            segments = {
                'champions': len(rfm[(r_quartiles == 4) & (f_quartiles == 4) & (m_quartiles == 4)]),
                'loyal': len(rfm[(r_quartiles >= 3) & (f_quartiles >= 3) & (m_quartiles >= 3)]),
                'risk': len(rfm[(r_quartiles == 1) & (f_quartiles >= 2)]),
                'lost': len(rfm[(r_quartiles == 1) & (f_quartiles == 1)])
            }
            
            return segments
    except Exception:
        return None

def generate_business_recommendations(metrics):
    """Generate detailed business recommendations based on metrics"""
    recommendations = []
    
    if metrics.get('sales'):
        sales = metrics['sales']
        if sales['growth_rate'] < 0:
            recommendations.append({
                "title": "Revenue Decline Alert",
                "description": "Sales are showing a negative trend. This could be due to:\n"
                             "• Market competition\n"
                             "• Pricing issues\n"
                             "• Product lifecycle changes\n"
                             "• Seasonal factors",
                "action": "Strategic Actions Required:\n"
                         "1. Review pricing strategy against competitors\n"
                         "2. Analyze customer feedback and pain points\n"
                         "3. Launch targeted promotional campaigns\n"
                         "4. Evaluate product portfolio performance",
                "theory": "According to the Price Elasticity Theory, adjusting prices can significantly "
                         "impact demand. Consider price optimization based on elasticity analysis."
            })
    
    if metrics.get('products'):
        products = metrics['products']
        recommendations.append({
            "title": "Product Portfolio Optimization",
            "description": "Analysis of product performance shows opportunities for optimization:\n"
                         "• Clear distinction between top and bottom performers\n"
                         "• Potential for product mix improvement",
            "action": "Recommended Actions:\n"
                     "1. Focus marketing on top-performing products\n"
                     "2. Review and potentially discontinue bottom performers\n"
                     "3. Analyze successful product characteristics\n"
                     "4. Consider product bundling strategies",
            "theory": "Based on the Pareto Principle (80/20 rule), focus resources on the top 20% "
                     "of products that likely generate 80% of revenue."
        })
    
    if metrics.get('customer'):
        customer = metrics['customer']
        if customer['retention_rate'] < 70:
            recommendations.append({
                "title": "Customer Retention Strategy",
                "description": f"Current retention rate at {customer['retention_rate']:.1f}% indicates "
                             "need for improved customer relationship management",
                "action": "Customer Retention Strategy:\n"
                         "1. Implement loyalty program\n"
                         "2. Develop personalized marketing campaigns\n"
                         "3. Regular customer feedback collection\n"
                         "4. Enhance customer service quality",
                "theory": "According to the Customer Lifetime Value (CLV) model, increasing retention "
                         "rates by 5% can increase profits by 25-95%. Focus on relationship marketing."
            })
    
    return recommendations

def generate_data_report(data):
    """Generate comprehensive data analysis report"""
    try:
        report_id = str(uuid.uuid4())
        print(f"Starting report generation with ID: {report_id}")
        
        df = pd.DataFrame(data)
        df = df.replace({np.nan: None, 'NaN': None, 'nan': None})
        
        # Detect data context
        context = detect_data_context(df)
        print(f"Detected context: {context}")
        
        # Generate base overview
        overview = {
            "recordCount": int(len(df)),
            "columnCount": int(len(df.columns)),
            "dataTypes": {str(k): int(v) for k, v in df.dtypes.value_counts().items()},
            "dateRange": get_date_range(df) or "No date range available",
            "memoryUsage": float(df.memory_usage().sum() / 1024 / 1024),
            "context": context
        }
        
        print("Generated overview section")
        
        # Generate statistics
        statistics = convert_to_native_types({
            "numerical": get_numerical_stats(df),
            "categorical": get_categorical_stats(df),
            "correlations": get_correlations(df)
        })
        
        print("Generated statistics section")
        
        # Generate quality metrics first
        quality = {
            "missingValues": {str(k): int(v) for k, v in df.isnull().sum().items()},
            "duplicates": int(df.duplicated().sum()),
            "uniqueValues": {str(k): int(v) for k, v in df.nunique().items()}
        }
        
        print("Generated quality metrics")
        
        # Generate insights after quality metrics are available
        generic_insights = generate_insights(df, statistics, quality)
        context_insights = generate_context_insights(df, context, statistics)
        all_insights = generic_insights + context_insights

        # Analyze trends
        trends = analyze_trends(df, context)
        
        # Generate recommendations
        recommendations = generate_context_recommendations(context, statistics, trends)

        # Enhanced business analytics for business context
        if context == "business":
            business_metrics = generate_business_metrics(df)
            business_recommendations = generate_business_recommendations(business_metrics)
            report = {
                "id": report_id,
                "title": f"{context.title()} Data Analysis Report",
                "timestamp": datetime.now().isoformat(),
                "overview": overview,
                "statistics": statistics,
                "quality": quality,
                "context_type": context,
                "recommendations": convert_to_native_types(recommendations),
                "insights": convert_to_native_types(all_insights),
                "visualizations": [],
                "business_metrics": convert_to_native_types(business_metrics),
                "business_recommendations": business_recommendations
            }
        else:
            report = {
                "id": report_id,
                "title": f"{context.title()} Data Analysis Report",
                "timestamp": datetime.now().isoformat(),
                "overview": overview,
                "statistics": statistics,
                "quality": quality,
                "context_type": context,
                "recommendations": convert_to_native_types(recommendations),
                "insights": convert_to_native_types(all_insights),
                "visualizations": []
            }

        # Store report
        REPORTS_STORAGE[report_id] = convert_to_native_types(report)
        print(f"Successfully stored report {report_id}")
        return report

    except Exception as e:
        print(f"Error generating report: {str(e)}")
        return None

@app.route('/analyze', methods=['POST'])
def analyze_data():
    try:
        report = request.json
        
        # Convert report data to string representation for Gemini
        analysis_data = {
            'records': report['overview']['recordCount'],
            'columns': report['overview']['columnCount'],
            'data_types': report['overview']['dataTypes'],
            'quality': report['quality']['missingValues'],
            'stats': report['statistics']
        }
        
        analysis_prompt = f"""
        As a data analyst, analyze this dataset and provide actionable recommendations:
        Dataset Overview:
        - Records: {analysis_data['records']}
        - Columns: {analysis_data['columns']}
        - Types: {json.dumps(analysis_data['data_types'])}
        
        Focus Areas:
        1. Data Quality: Identify and suggest fixes for data quality issues
        2. Analysis Potential: Recommend analysis approaches
        3. Visualization: Suggest effective visualization methods
        4. Business Impact: Identify potential business insights
        5. Improvements: Suggest data collection/structure improvements

        Format response as JSON:
        {{
            "recommendations": [
                {{
                    "title": "Clear title of recommendation",
                    "description": "Detailed explanation",
                    "action": "Specific action steps"
                }}
            ]
        }}
        Only return valid JSON, no other text.
        """

        response = http.post(
            f"{GEMINI_URL}?key={GEMINI_API_KEY}",
            headers={'Content-Type': 'application/json'},
            json={"contents": [{"parts": [{"text": analysis_prompt}]}]},
            timeout=15
        )
        
        if response.status_code != 200:
            return jsonify({
                "recommendations": [{
                    "title": "Analysis Error",
                    "description": "Could not generate AI recommendations",
                    "action": "Please try again later"
                }]
            })
        
        try:
            result = response.json()
            text_response = result['candidates'][0]['content']['parts'][0]['text']
            recommendations = json.loads(text_response)
            return jsonify(recommendations)
        except:
            # Fallback recommendations if JSON parsing fails
            return jsonify({
                "recommendations": [{
                    "title": "Basic Data Overview",
                    "description": f"Dataset contains {analysis_data['records']} records across {analysis_data['columns']} columns",
                    "action": "Review the data quality metrics and column distributions"
                }]
            })
        
    except Exception as e:
        print(f"Analysis error: {str(e)}")
        return jsonify({
            "recommendations": [{
                "title": "Analysis Not Available",
                "description": "There was an error analyzing the data",
                "action": "Check the data format and try again"
            }]
        })

def get_date_range(df):
    """Get date range from dataframe"""
    date_cols = df.select_dtypes(include=['datetime64']).columns
    if not len(date_cols):
        # Try common date formats
        date_formats = [
            '%Y-%m-%d',
            '%d/%m/%Y',
            '%m/%d/%Y',
            '%Y/%m/%d',
            '%d-%m-%Y',
            '%m-%d-%Y'
        ]
        
        for col in df.select_dtypes(include=['object']):
            for date_format in date_formats:
                try:
                    dates = pd.to_datetime(df[col], format=date_format, errors='coerce')
                    if not dates.isnull().all():
                        return f"{dates.min():%Y-%m-%d} to {dates.max():%Y-%m-%d}"
                except:
                    continue
    elif len(date_cols):
        dates = df[date_cols[0]]
        return f"{dates.min():%Y-%m-%d} to {dates.max():%Y-%m-%d}"
    return "No date columns found"

def get_categorical_stats(df):
    """Get statistics for categorical columns"""
    cat_cols = df.select_dtypes(include=['object']).columns
    stats_dict = {}
    
    for col in cat_cols:
        value_counts = df[col].value_counts()
        stats_dict[col] = {
            "unique_values": len(value_counts),
            "top_values": value_counts.head(5).to_dict(),
            "null_count": df[col].isnull().sum()
        }
    
    return stats_dict

def get_correlations(df):
    """Calculate correlations between numeric columns"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        return corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)).dropna(how='all').to_dict()
    return {}

def generate_chart_data(df, chart_config):
    """Generate chart configuration based on chart type"""
    try:
        chart_type = chart_config['type']
        if chart_type == 'bar':
            return {
                'type': 'bar',
                'data': {
                    'labels': df[chart_config['suggested_columns'][0]].tolist(),
                    'datasets': [{
                        'label': chart_config['suggested_columns'][1],
                        'data': df[chart_config['suggested_columns'][1]].tolist(),
                        'backgroundColor': 'rgba(59, 130, 246, 0.5)'
                    }]
                }
            }
        elif chart_type == 'line':
            return {
                'type': 'line',
                'data': {
                    'labels': df[chart_config['suggested_columns']['x']].tolist(),
                    'datasets': [{
                        'label': chart_config['suggested_columns']['y'],
                        'data': df[chart_config['suggested_columns']['y']].tolist(),
                        'borderColor': 'rgba(59, 130, 246, 1)',
                        'fill': False
                    }]
                }
            }
        # Add other chart types as needed
        return None
    except Exception as e:
        print(f"Error generating chart data: {e}")
        return None

def generate_insights(df, statistics, quality):
    """Generate insights from data analysis with error handling"""
    insights = []
    
    try:
        # Data completeness insights
        if quality and quality.get('missingValues'):
            null_counts = quality['missingValues']
            if any(null_counts.values()):
                problem_cols = [col for col, count in null_counts.items() if count > 0]
                insights.append({
                    "type": "data_quality",
                    "severity": "medium" if max(null_counts.values()) / len(df) < 0.1 else "high",
                    "message": f"Missing values detected in columns: {', '.join(problem_cols)}"
                })

        # Statistical insights
        if statistics and statistics.get('numerical'):
            for col, stats in statistics['numerical'].items():
                try:
                    skewness = stats.get('skewness')
                    if skewness is not None and abs(float(skewness)) > 2:
                        insights.append({
                            "type": "statistical",
                            "severity": "medium",
                            "message": f"Column '{col}' shows significant skewness ({float(skewness):.2f})"
                        })
                except (TypeError, ValueError) as e:
                    print(f"Error processing skewness for column {col}: {str(e)}")
                    continue

        # Correlation insights
        if statistics and statistics.get('correlations'):
            strong_correlations = []
            for col1, corrs in statistics['correlations'].items():
                for col2, corr in corrs.items():
                    try:
                        if corr is not None and abs(float(corr)) > 0.7:
                            strong_correlations.append((col1, col2, corr))
                    except (TypeError, ValueError):
                        continue
            
            if strong_correlations:
                insights.append({
                    "type": "correlation",
                    "severity": "info",
                    "message": f"Strong correlations found between {len(strong_correlations)} column pairs"
                })

    except Exception as e:
        print(f"Error generating insights: {str(e)}")
        insights.append({
            "type": "error",
            "severity": "high",
            "message": "Error analyzing data relationships"
        })

    return insights

def get_numerical_stats(df):
    """Get statistics for numerical columns with error handling"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    stats_dict = {}
    
    for col in numeric_cols:
        try:
            clean_data = df[col].dropna()
            if len(clean_data) == 0:
                continue
                
            stats_dict[col] = {
                "mean": float(clean_data.mean()),
                "median": float(clean_data.median()),
                "std": float(clean_data.std()),
                "min": float(clean_data.min()),
                "max": float(clean_data.max()),
                "skewness": float(stats.skew(clean_data)),
                "kurtosis": float(stats.kurtosis(clean_data))
            }
        except Exception as e:
            print(f"Error calculating stats for column {col}: {str(e)}")
            continue
    
    return stats_dict

if __name__ == '__main__':
    success = False
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            if init_db():
                success = True
                break
        except Exception as e:
            if attempt < max_attempts - 1:
                time.sleep(1)
                continue
            else:
                print(f"Failed to initialize database after {max_attempts} attempts: {str(e)}")
                raise
    
    if success:
        app.run(debug=True)
    else:
        print("Could not initialize the application")
