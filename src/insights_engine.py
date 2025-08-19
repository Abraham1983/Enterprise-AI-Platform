# Insights and Analytics Engine - KPIs, Trends, Forecasts, Anomalies
# Powered by scikit-learn for anomaly detection and statistical analysis

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
from collections import defaultdict
import sqlite3

# Machine learning for anomaly detection
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import joblib

# Database
from sqlalchemy import text
from sqlalchemy.orm import Session

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class InsightsConfig:
    cache_ttl_hours: int = 24
    anomaly_threshold: float = 0.1  # 10% outliers
    forecast_days: int = 90
    min_data_points: int = 10
    enable_ml_anomalies: bool = True
    enable_forecasting: bool = True

class InsightsEngine:
    """Main analytics and insights engine"""
    
    def __init__(self, db_session: Session, config: InsightsConfig = None):
        self.db = db_session
        self.config = config or InsightsConfig()
        self.cache = {}
        self.cache_timestamps = {}
        
        # Initialize ML models
        self.anomaly_model = None
        self.scaler = StandardScaler()
        self._load_or_create_models()
    
    def _load_or_create_models(self):
        """Load or create ML models for anomaly detection"""
        try:
            # Try to load existing model
            if os.path.exists('models/anomaly_model.joblib'):
                self.anomaly_model = joblib.load('models/anomaly_model.joblib')
                self.scaler = joblib.load('models/scaler.joblib')
                logger.info("Loaded existing anomaly detection models")
            else:
                # Create new model
                self.anomaly_model = IsolationForest(
                    contamination=self.config.anomaly_threshold,
                    random_state=42,
                    n_estimators=100
                )
                logger.info("Created new anomaly detection model")
        except Exception as e:
            logger.error(f"Failed to load/create ML models: {e}")
            self.anomaly_model = None
    
    def _save_models(self):
        """Save trained ML models"""
        try:
            os.makedirs('models', exist_ok=True)
            if self.anomaly_model:
                joblib.dump(self.anomaly_model, 'models/anomaly_model.joblib')
                joblib.dump(self.scaler, 'models/scaler.joblib')
                logger.info("Saved ML models")
        except Exception as e:
            logger.error(f"Failed to save ML models: {e}")
    
    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached data is still valid"""
        if key not in self.cache_timestamps:
            return False
        
        cache_time = self.cache_timestamps[key]
        ttl = timedelta(hours=self.config.cache_ttl_hours)
        return datetime.utcnow() - cache_time < ttl
    
    def _get_cached_or_compute(self, key: str, compute_func, *args, **kwargs):
        """Get cached result or compute and cache"""
        if self._is_cache_valid(key):
            return self.cache[key]
        
        result = compute_func(*args, **kwargs)
        self.cache[key] = result
        self.cache_timestamps[key] = datetime.utcnow()
        return result
    
    def get_invoice_data(self, days_back: int = 365) -> pd.DataFrame:
        """Get invoice data as DataFrame for analysis"""
        try:
            # This assumes you have invoice data in your database
            # Adjust the query based on your actual schema
            query = text("""
                SELECT 
                    invoice_number,
                    client_name,
                    client_email,
                    total_amount,
                    tax_amount,
                    currency,
                    status,
                    created_at,
                    sent_at,
                    paid_at,
                    due_date,
                    project_name,
                    hours_worked,
                    hourly_rate
                FROM invoices 
                WHERE created_at >= :start_date
                ORDER BY created_at DESC
            """)
            
            start_date = datetime.utcnow() - timedelta(days=days_back)
            result = self.db.execute(query, {"start_date": start_date})
            
            # Convert to DataFrame
            columns = [
                'invoice_number', 'client_name', 'client_email', 'total_amount',
                'tax_amount', 'currency', 'status', 'created_at', 'sent_at',
                'paid_at', 'due_date', 'project_name', 'hours_worked', 'hourly_rate'
            ]
            
            data = []
            for row in result:
                data.append(dict(zip(columns, row)))
            
            df = pd.DataFrame(data)
            
            # Convert date columns
            date_columns = ['created_at', 'sent_at', 'paid_at', 'due_date']
            for col in date_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])
            
            # Calculate derived metrics
            if not df.empty:
                df['days_to_pay'] = (df['paid_at'] - df['sent_at']).dt.days
                df['is_overdue'] = df['due_date'] < datetime.utcnow()
                df['aging_days'] = (datetime.utcnow() - df['due_date']).dt.days
                df['aging_days'] = df['aging_days'].clip(lower=0)
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to get invoice data: {e}")
            return pd.DataFrame()
    
    def get_payment_data(self, days_back: int = 365) -> pd.DataFrame:
        """Get payment data as DataFrame"""
        try:
            query = text("""
                SELECT 
                    invoice_number,
                    payment_method,
                    status,
                    amount_usd,
                    currency,
                    created_at,
                    paid_at,
                    expires_at
                FROM payments 
                WHERE created_at >= :start_date
                ORDER BY created_at DESC
            """)
            
            start_date = datetime.utcnow() - timedelta(days=days_back)
            result = self.db.execute(query, {"start_date": start_date})
            
            columns = [
                'invoice_number', 'payment_method', 'status', 'amount_usd',
                'currency', 'created_at', 'paid_at', 'expires_at'
            ]
            
            data = []
            for row in result:
                data.append(dict(zip(columns, row)))
            
            df = pd.DataFrame(data)
            
            # Convert date columns
            date_columns = ['created_at', 'paid_at', 'expires_at']
            for col in date_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to get payment data: {e}")
            return pd.DataFrame()
    
    def compute_kpis(self) -> Dict[str, Any]:
        """Compute key performance indicators"""
        
        def _compute():
            df = self.get_invoice_data()
            payments_df = self.get_payment_data()
            
            if df.empty:
                return {
                    'total_invoices': 0,
                    'total_revenue': 0,
                    'avg_invoice_amount': 0,
                    'paid_invoices': 0,
                    'payment_rate': 0,
                    'avg_days_to_pay': 0,
                    'overdue_invoices': 0,
                    'overdue_amount': 0,
                    'mrr': 0,
                    'arr': 0
                }
            
            # Basic metrics
            total_invoices = len(df)
            total_revenue = df['total_amount'].sum()
            avg_invoice_amount = df['total_amount'].mean()
            
            # Payment metrics
            paid_df = df[df['status'] == 'paid']
            paid_invoices = len(paid_df)
            payment_rate = paid_invoices / total_invoices if total_invoices > 0 else 0
            
            avg_days_to_pay = paid_df['days_to_pay'].mean() if not paid_df.empty else 0
            
            # Overdue metrics
            overdue_df = df[df['is_overdue'] & (df['status'] != 'paid')]
            overdue_invoices = len(overdue_df)
            overdue_amount = overdue_df['total_amount'].sum()
            
            # Recurring revenue (simplified - assumes monthly billing)
            current_month = datetime.utcnow().replace(day=1)
            monthly_df = df[df['created_at'] >= current_month]
            mrr = monthly_df['total_amount'].sum()
            arr = mrr * 12
            
            return {
                'total_invoices': int(total_invoices),
                'total_revenue': float(total_revenue),
                'avg_invoice_amount': float(avg_invoice_amount),
                'paid_invoices': int(paid_invoices),
                'payment_rate': float(payment_rate),
                'avg_days_to_pay': float(avg_days_to_pay) if not np.isnan(avg_days_to_pay) else 0,
                'overdue_invoices': int(overdue_invoices),
                'overdue_amount': float(overdue_amount),
                'mrr': float(mrr),
                'arr': float(arr),
                'computed_at': datetime.utcnow().isoformat()
            }
        
        return self._get_cached_or_compute('kpis', _compute)
    
    def compute_ar_aging(self) -> Dict[str, Any]:
        """Compute accounts receivable aging"""
        
        def _compute():
            df = self.get_invoice_data()
            
            if df.empty:
                return {
                    'current': 0,
                    '1_30_days': 0,
                    '31_60_days': 0,
                    '61_90_days': 0,
                    'over_90_days': 0,
                    'total_ar': 0
                }
            
            # Filter unpaid invoices
            unpaid_df = df[df['status'] != 'paid']
            
            if unpaid_df.empty:
                return {
                    'current': 0,
                    '1_30_days': 0,
                    '31_60_days': 0,
                    '61_90_days': 0,
                    'over_90_days': 0,
                    'total_ar': 0
                }
            
            # Categorize by aging
            current = unpaid_df[unpaid_df['aging_days'] <= 0]['total_amount'].sum()
            days_1_30 = unpaid_df[(unpaid_df['aging_days'] > 0) & (unpaid_df['aging_days'] <= 30)]['total_amount'].sum()
            days_31_60 = unpaid_df[(unpaid_df['aging_days'] > 30) & (unpaid_df['aging_days'] <= 60)]['total_amount'].sum()
            days_61_90 = unpaid_df[(unpaid_df['aging_days'] > 60) & (unpaid_df['aging_days'] <= 90)]['total_amount'].sum()
            over_90 = unpaid_df[unpaid_df['aging_days'] > 90]['total_amount'].sum()
            
            total_ar = unpaid_df['total_amount'].sum()
            
            return {
                'current': float(current),
                '1_30_days': float(days_1_30),
                '31_60_days': float(days_31_60),
                '61_90_days': float(days_61_90),
                'over_90_days': float(over_90),
                'total_ar': float(total_ar),
                'computed_at': datetime.utcnow().isoformat()
            }
        
        return self._get_cached_or_compute('ar_aging', _compute)
    
    def compute_trends(self, months_back: int = 12) -> Dict[str, Any]:
        """Compute revenue and payment trends"""
        
        def _compute():
            df = self.get_invoice_data(days_back=months_back * 30)
            
            if df.empty:
                return {
                    'monthly_revenue': [],
                    'monthly_invoices': [],
                    'payment_trends': [],
                    'client_trends': []
                }
            
            # Monthly revenue trend
            df['month'] = df['created_at'].dt.to_period('M')
            monthly_revenue = df.groupby('month')['total_amount'].sum().reset_index()
            monthly_revenue['month'] = monthly_revenue['month'].astype(str)
            
            # Monthly invoice count
            monthly_invoices = df.groupby('month').size().reset_index(name='count')
            monthly_invoices['month'] = monthly_invoices['month'].astype(str)
            
            # Payment method trends
            payments_df = self.get_payment_data(days_back=months_back * 30)
            if not payments_df.empty:
                payments_df['month'] = payments_df['created_at'].dt.to_period('M')
                payment_trends = payments_df.groupby(['month', 'payment_method']).size().reset_index(name='count')
                payment_trends['month'] = payment_trends['month'].astype(str)
            else:
                payment_trends = pd.DataFrame()
            
            # Top clients by revenue
            client_trends = df.groupby('client_name')['total_amount'].sum().sort_values(ascending=False).head(10).reset_index()
            
            return {
                'monthly_revenue': monthly_revenue.to_dict('records'),
                'monthly_invoices': monthly_invoices.to_dict('records'),
                'payment_trends': payment_trends.to_dict('records') if not payment_trends.empty else [],
                'client_trends': client_trends.to_dict('records'),
                'computed_at': datetime.utcnow().isoformat()
            }
        
        return self._get_cached_or_compute('trends', _compute)
    
    def detect_anomalies(self) -> Dict[str, Any]:
        """Detect anomalies in invoice and payment data"""
        
        def _compute():
            df = self.get_invoice_data()
            
            if df.empty or len(df) < self.config.min_data_points:
                return {
                    'anomalies': [],
                    'summary': {
                        'total_anomalies': 0,
                        'high_amount_anomalies': 0,
                        'timing_anomalies': 0,
                        'rate_anomalies': 0
                    }
                }
            
            anomalies = []
            
            # 1. Amount-based anomalies using IsolationForest
            if self.config.enable_ml_anomalies and self.anomaly_model:
                try:
                    # Prepare features for ML model
                    features = []
                    valid_indices = []
                    
                    for idx, row in df.iterrows():
                        if pd.notna(row['total_amount']) and pd.notna(row['hours_worked']) and pd.notna(row['hourly_rate']):
                            features.append([
                                row['total_amount'],
                                row['hours_worked'] if row['hours_worked'] else 0,
                                row['hourly_rate'] if row['hourly_rate'] else 0,
                                row['tax_amount'] if row['tax_amount'] else 0
                            ])
                            valid_indices.append(idx)
                    
                    if len(features) >= self.config.min_data_points:
                        features_array = np.array(features)
                        features_scaled = self.scaler.fit_transform(features_array)
                        
                        # Fit and predict anomalies
                        self.anomaly_model.fit(features_scaled)
                        anomaly_scores = self.anomaly_model.decision_function(features_scaled)
                        anomaly_labels = self.anomaly_model.predict(features_scaled)
                        
                        # Save the trained model
                        self._save_models()
                        
                        # Identify anomalies
                        for i, (idx, score, label) in enumerate(zip(valid_indices, anomaly_scores, anomaly_labels)):
                            if label == -1:  # Anomaly
                                row = df.iloc[idx]
                                anomalies.append({
                                    'type': 'ml_anomaly',
                                    'invoice_number': row['invoice_number'],
                                    'client_name': row['client_name'],
                                    'amount': float(row['total_amount']),
                                    'anomaly_score': float(score),
                                    'reason': 'Statistical outlier detected by ML model',
                                    'severity': 'high' if score < -0.5 else 'medium',
                                    'created_at': row['created_at'].isoformat() if pd.notna(row['created_at']) else None
                                })
                
                except Exception as e:
                    logger.error(f"ML anomaly detection failed: {e}")
            
            # 2. Rule-based anomalies
            
            # High amount anomalies (> 3 standard deviations)
            if len(df) > 5:
                amount_mean = df['total_amount'].mean()
                amount_std = df['total_amount'].std()
                high_threshold = amount_mean + (3 * amount_std)
                
                high_amount_df = df[df['total_amount'] > high_threshold]
                for _, row in high_amount_df.iterrows():
                    anomalies.append({
                        'type': 'high_amount',
                        'invoice_number': row['invoice_number'],
                        'client_name': row['client_name'],
                        'amount': float(row['total_amount']),
                        'threshold': float(high_threshold),
                        'reason': f'Amount ${row["total_amount"]:.2f} exceeds threshold ${high_threshold:.2f}',
                        'severity': 'high',
                        'created_at': row['created_at'].isoformat() if pd.notna(row['created_at']) else None
                    })
            
            # Duplicate detection
            duplicates = df[df.duplicated(['client_name', 'total_amount', 'project_name'], keep=False)]
            for _, row in duplicates.iterrows():
                anomalies.append({
                    'type': 'potential_duplicate',
                    'invoice_number': row['invoice_number'],
                    'client_name': row['client_name'],
                    'amount': float(row['total_amount']),
                    'reason': 'Potential duplicate invoice (same client, amount, project)',
                    'severity': 'medium',
                    'created_at': row['created_at'].isoformat() if pd.notna(row['created_at']) else None
                })
            
            # Summary
            summary = {
                'total_anomalies': len(anomalies),
                'high_amount_anomalies': len([a for a in anomalies if a['type'] == 'high_amount']),
                'timing_anomalies': len([a for a in anomalies if a['type'] == 'weekend_work']),
                'rate_anomalies': len([a for a in anomalies if a['type'] == 'rate_anomaly']),
                'ml_anomalies': len([a for a in anomalies if a['type'] == 'ml_anomaly']),
                'duplicates': len([a for a in anomalies if a['type'] == 'potential_duplicate'])
            }
            
            return {
                'anomalies': anomalies,
                'summary': summary,
                'computed_at': datetime.utcnow().isoformat()
            }
        
        return self._get_cached_or_compute('anomalies', _compute)
    
    def forecast_cash_flow(self, days_ahead: int = 90) -> Dict[str, Any]:
        """Forecast cash flow based on historical payment patterns"""
        
        def _compute():
            df = self.get_invoice_data()
            
            if df.empty or not self.config.enable_forecasting:
                return {
                    'forecast': [],
                    'total_expected': 0,
                    'confidence': 'low'
                }
            
            # Get unpaid invoices
            unpaid_df = df[df['status'] != 'paid'].copy()
            
            if unpaid_df.empty:
                return {
                    'forecast': [],
                    'total_expected': 0,
                    'confidence': 'high'
                }
            
            # Calculate historical payment patterns by client
            paid_df = df[df['status'] == 'paid'].copy()
            
            if paid_df.empty:
                # No historical data, use default assumptions
                payment_probability = 0.8
                avg_days_to_pay = 30
            else:
                # Calculate client-specific payment patterns
                client_patterns = {}
                for client in paid_df['client_name'].unique():
                    client_paid = paid_df[paid_df['client_name'] == client]
                    avg_days = client_paid['days_to_pay'].mean()
                    payment_rate = len(client_paid) / len(df[df['client_name'] == client])
                    
                    client_patterns[client] = {
                        'avg_days_to_pay': avg_days if not np.isnan(avg_days) else 30,
                        'payment_probability': min(payment_rate, 1.0)
                    }
                
                # Overall averages
                payment_probability = paid_df['days_to_pay'].count() / len(df)
                avg_days_to_pay = paid_df['days_to_pay'].mean()
            
            # Generate forecast
            forecast = []
            total_expected = 0
            
            for _, invoice in unpaid_df.iterrows():
                client = invoice['client_name']
                
                # Use client-specific patterns if available
                if client in client_patterns:
                    prob = client_patterns[client]['payment_probability']
                    days = client_patterns[client]['avg_days_to_pay']
                else:
                    prob = payment_probability
                    days = avg_days_to_pay if not np.isnan(avg_days_to_pay) else 30
                
                # Estimate payment date
                if pd.notna(invoice['sent_at']):
                    expected_date = invoice['sent_at'] + timedelta(days=int(days))
                else:
                    expected_date = datetime.utcnow() + timedelta(days=int(days))
                
                # Only include if within forecast window
                if expected_date <= datetime.utcnow() + timedelta(days=days_ahead):
                    expected_amount = invoice['total_amount'] * prob
                    total_expected += expected_amount
                    
                    forecast.append({
                        'invoice_number': invoice['invoice_number'],
                        'client_name': client,
                        'amount': float(invoice['total_amount']),
                        'expected_amount': float(expected_amount),
                        'probability': float(prob),
                        'expected_date': expected_date.isoformat(),
                        'days_overdue': int(invoice['aging_days']) if pd.notna(invoice['aging_days']) else 0
                    })
            
            # Sort by expected date
            forecast.sort(key=lambda x: x['expected_date'])
            
            # Determine confidence based on data quality
            confidence = 'high' if len(paid_df) > 20 else 'medium' if len(paid_df) > 5 else 'low'
            
            return {
                'forecast': forecast,
                'total_expected': float(total_expected),
                'confidence': confidence,
                'forecast_period_days': days_ahead,
                'computed_at': datetime.utcnow().isoformat()
            }
        
        return self._get_cached_or_compute('forecast', _compute)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive insights summary"""
        
        return {
            'kpis': self.compute_kpis(),
            'ar_aging': self.compute_ar_aging(),
            'trends': self.compute_trends(),
            'anomalies': self.detect_anomalies(),
            'forecast': self.forecast_cash_flow(),
            'generated_at': datetime.utcnow().isoformat()
        }

# Background job functions
def run_nightly_insights(db_session: Session, config: InsightsConfig = None):
    """Run nightly insights computation and caching"""
    
    logger.info("Starting nightly insights computation")
    
    try:
        engine = InsightsEngine(db_session, config)
        
        # Compute and cache all insights
        engine.compute_kpis()
        engine.compute_ar_aging()
        engine.compute_trends()
        engine.detect_anomalies()
        engine.forecast_cash_flow()
        
        logger.info("Nightly insights computation completed successfully")
        
    except Exception as e:
        logger.error(f"Nightly insights computation failed: {e}")
        raise

def run_anomaly_scan(db_session: Session, config: InsightsConfig = None):
    """Run anomaly detection scan"""
    
    logger.info("Starting anomaly scan")
    
    try:
        engine = InsightsEngine(db_session, config)
        anomalies = engine.detect_anomalies()
        
        # Log high-severity anomalies
        high_severity = [a for a in anomalies['anomalies'] if a['severity'] == 'high']
        if high_severity:
            logger.warning(f"Found {len(high_severity)} high-severity anomalies")
            for anomaly in high_severity:
                logger.warning(f"Anomaly: {anomaly['type']} - {anomaly['reason']}")
        
        logger.info(f"Anomaly scan completed. Found {anomalies['summary']['total_anomalies']} total anomalies")
        
        return anomalies
        
    except Exception as e:
        logger.error(f"Anomaly scan failed: {e}")
        raise

# Example usage
if __name__ == "__main__":
    # This would be used with your actual database session
    print("Insights Engine initialized with ML-powered anomaly detection!")