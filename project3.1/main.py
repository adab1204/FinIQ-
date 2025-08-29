# FinIQ - Advanced Financial Intelligence & Analytics Platform

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from datetime import datetime, timedelta
import warnings
import io
import time
warnings.filterwarnings('ignore')

# Page configuration with professional theme
st.set_page_config(
    page_title="FinIQ - Professional Finance Analytics",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Professional CSS with sleek, professional styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Source+Sans+Pro:wght@300;400;600;700&display=swap');

/* Global font and color improvements */
html, body, [class*="css"] {
    font-family: 'Inter', 'Source Sans Pro', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #e2e8f0 !important;
}

/* Enhanced sidebar styling with better text visibility */
.css-1d391kg {
    background: linear-gradient(180deg, #2d3748 0%, #1a202c 100%);
    border-right: 2px solid #4a5568;
}

/* Professional sidebar title with proper diamond icon */
.sidebar-title {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 12px;
    font-family: 'Inter', sans-serif;
    font-size: 28px;
    font-weight: 700;
    color: #667eea;
    padding: 20px 0;
    margin-bottom: 24px;
    border-bottom: 3px solid #667eea;
    text-align: center;
}

/* FIXED: Lightened chart title color for better visibility */
.chart-title {
    text-align: center;
    font-family: 'Inter', sans-serif;
    font-size: 16px;
    font-weight: 600;
    color: #94a3b8 !important;
    margin-top: -5px;
    margin-bottom: 20px;
    padding: 8px 0;
    border-bottom: 1px solid #e2e8f0;
}

/* Center-aligned Data Management section */
.sidebar-section-header {
    text-align: center;
    font-family: 'Inter', sans-serif;
    font-size: 16px;
    font-weight: 600;
    color: #cbd5e0 !important;
    margin-bottom: 16px;
    padding: 10px 0;
    border-bottom: 1px solid #4a5568;
}

/* Improved sidebar text visibility */
.css-1d391kg .stMarkdown {
    color: #e2e8f0 !important;
}

.css-1d391kg .stMarkdown h3 {
    color: #f7fafc !important;
    font-weight: 600 !important;
    text-align: center !important;
}

.css-1d391kg .stMarkdown p {
    color: #cbd5e0 !important;
    font-size: 14px !important;
}

/* Professional main header */
.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2.5rem 2rem;
    border-radius: 16px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0 20px 40px rgba(102, 126, 234, 0.3);
}

.main-header h1 {
    font-family: 'Inter', sans-serif;
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
    letter-spacing: -1px;
    color: #ffffff !important;
}

.main-header p {
    font-family: 'Source Sans Pro', sans-serif;
    font-size: 1.1rem;
    font-weight: 400;
    opacity: 0.95;
    color: #ffffff !important;
}

/* Enhanced metric cards with better visibility */
.metric-container {
    background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
    padding: 1.5rem;
    border-radius: 12px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
    border: 1px solid #e2e8f0;
    margin-bottom: 1rem;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.metric-container:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.12);
}

.metric-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.8rem;
    font-weight: 600;
    color: #4a5568 !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 0.5rem;
}

.metric-value {
    font-family: 'Inter', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: #1a202c !important;
    margin-bottom: 0.25rem;
}

.metric-delta {
    font-family: 'Source Sans Pro', sans-serif;
    font-size: 0.75rem;
    font-weight: 500;
}

/* Enhanced cards for insights and recommendations */
.insight-card {
    background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%);
    padding: 1.8rem;
    border-radius: 16px;
    margin-bottom: 1.2rem;
    border-left: 5px solid #667eea;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    transition: all 0.3s ease;
}

.insight-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 16px 48px rgba(0, 0, 0, 0.15);
}

.insight-card h4 {
    font-family: 'Inter', sans-serif;
    font-size: 1.1rem;
    font-weight: 600;
    color: #2d3748 !important;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
}

.insight-card p {
    font-family: 'Source Sans Pro', sans-serif;
    font-size: 0.95rem;
    line-height: 1.6;
    color: #4a5568 !important;
    margin: 0;
}

.recommendation-card {
    background: linear-gradient(145deg, #f0fff4 0%, #f7fafc 100%);
    padding: 1.8rem;
    border-radius: 16px;
    margin-bottom: 1.2rem;
    border-left: 5px solid #48bb78;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    transition: all 0.3s ease;
}

.recommendation-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 16px 48px rgba(0, 0, 0, 0.15);
}

.recommendation-card h4 {
    font-family: 'Inter', sans-serif;
    font-size: 1.1rem;
    font-weight: 600;
    color: #2d3748 !important;
    margin-bottom: 1rem;
}

.recommendation-card p {
    font-family: 'Source Sans Pro', sans-serif;
    font-size: 0.95rem;
    line-height: 1.6;
    color: #2d3748 !important;
    margin: 0;
}

/* Anomaly card styling */
.anomaly-card {
    background: linear-gradient(145deg, #fff5f5 0%, #fed7d7 100%);
    padding: 1.8rem;
    border-radius: 16px;
    margin-bottom: 1.2rem;
    border-left: 5px solid #f56565;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    transition: all 0.3s ease;
}

.anomaly-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 16px 48px rgba(0, 0, 0, 0.15);
}

.anomaly-card h4 {
    font-family: 'Inter', sans-serif;
    font-size: 1.05rem;
    font-weight: 600;
    color: #c53030 !important;
    margin-bottom: 1rem;
}

.anomaly-card p {
    font-family: 'Source Sans Pro', sans-serif;
    font-size: 0.9rem;
    line-height: 1.6;
    color: #2d3748 !important;
    margin: 0;
}

/* Enhanced section headers */
.section-header {
    font-family: 'Inter', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: #ffffff !important;
    background: linear-gradient(135deg, #667eea, #764ba2);
    padding: 0.8rem 1.2rem;
    border-radius: 8px;
    margin-bottom: 1.5rem;
    text-align: center;
    box-shadow: 0 4px 16px rgba(102, 126, 234, 0.3);
}

/* Enhanced tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: #2d3748;
    padding: 6px;
    border-radius: 12px;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    font-size: 0.9rem;
    background-color: #4a5568;
    border-radius: 8px;
    padding: 0.6rem 1.2rem;
    border: 1px solid #718096;
    color: #e2e8f0 !important;
}

.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: #ffffff !important;
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
}

/* Professional form styling without emojis */
.form-header {
    font-family: 'Inter', sans-serif;
    font-size: 1.1rem;
    font-weight: 500;
    color: #cbd5e0 !important;
    margin-bottom: 1.5rem;
    text-align: center;
    letter-spacing: 0.5px;
}

/* Add Transaction form background styling */
div[data-testid="stExpander"] > div[data-testid="stExpanderDetails"] {
    background: linear-gradient(135deg, #2d3748 0%, #1a202c 100%) !important;
    padding: 25px !important;
    border-radius: 12px !important;
    border: 1px solid #4a5568 !important;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.15) !important;
}

/* Form elements within expander */
div[data-testid="stExpander"] .stForm {
    background: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid #4a5568 !important;
    backdrop-filter: blur(10px) !important;
}

/* Text color in form expander */
div[data-testid="stExpander"] .stMarkdown {
    color: #e2e8f0 !important;
}

div[data-testid="stExpander"] label {
    color: #cbd5e0 !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
}

/* Table styling */
.stDataFrame {
    font-family: 'Source Sans Pro', sans-serif;
    font-size: 0.9rem;
}

.stDataFrame td, .stDataFrame th {
    text-align: center !important;
    padding: 10px 14px !important;
    border-bottom: 1px solid #e2e8f0 !important;
    color: #2d3748 !important;
}

.stDataFrame th {
    font-weight: 600 !important;
    background-color: #f7fafc !important;
    color: #4a5568 !important;
}

/* Button improvements */
.stButton > button {
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    border-radius: 8px;
    transition: all 0.2s ease;
    border: none;
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white !important;
    font-size: 0.9rem;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #5a67d8, #6b46c1);
    transform: translateY(-1px);
    box-shadow: 0 8px 16px rgba(102, 126, 234, 0.3);
}

/* Sidebar styling improvements */
.css-1d391kg .stButton > button {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: #ffffff !important;
    border: none;
    font-weight: 600;
    font-size: 0.85rem;
}

.css-1d391kg .stButton > button:hover {
    background: linear-gradient(135deg, #5a67d8, #6b46c1);
    color: #ffffff !important;
}

/* Filter section improvements */
.css-1d391kg .stSelectbox label {
    color: #cbd5e0 !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
}

.css-1d391kg .stDateInput label {
    color: #cbd5e0 !important;
    font-weight: 500 !important;
}

.css-1d391kg .stMultiSelect label {
    color: #cbd5e0 !important;
    font-weight: 500 !important;
}

.css-1d391kg .stCheckbox label {
    color: #cbd5e0 !important;
    font-weight: 500 !important;
}

/* Professional text improvements */
.stMarkdown {
    font-family: 'Source Sans Pro', sans-serif;
    color: #2d3748 !important;
}

.stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
    font-family: 'Inter', sans-serif !important;
    color: #2d3748 !important;
}

/* Main content area background */
.main .block-container {
    background-color: #ffffff;
    color: #2d3748 !important;
}

/* Welcome section improvements */
.welcome-section h3 {
    color: #2d3748 !important;
    font-weight: 600 !important;
    margin-bottom: 1rem !important;
    font-size: 1.1rem !important;
}

.welcome-section p, .welcome-section li {
    color: #4a5568 !important;
    line-height: 1.5 !important;
    font-size: 0.9rem !important;
}

</style>
""", unsafe_allow_html=True)

class StreamlitFinanceAdvisor:
    def __init__(self):
        if 'advisor' not in st.session_state:
            st.session_state.advisor = self
        self.transactions = []
        self.spending_model = None
        self.anomaly_detector = None
        self.scaler = StandardScaler()
        self.label_encoders = {}

    def add_transaction(self, amount, category, description, date=None):
        """Add a new transaction to the database"""
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        transaction = {
            'date': date,
            'amount': amount,
            'category': category,
            'description': description,
            'day_of_week': datetime.strptime(date, '%Y-%m-%d').weekday(),
            'month': datetime.strptime(date, '%Y-%m-%d').month
        }
        self.transactions.append(transaction)

    def generate_sample_data(self, num_transactions=1000):
        """Generate realistic sample financial data for upper middle class Indian household"""
        # Categories typical for upper middle class Indian household
        categories = ['Groceries', 'Transportation', 'Entertainment', 'Utilities',
                     'Healthcare', 'Education', 'Dining Out', 'Clothing', 'Shopping', 'Income']
        
        # Fixed probability weights - ensuring more Income transactions
        category_weights = np.array([0.18, 0.10, 0.06, 0.08, 0.05, 0.06, 0.08, 0.05, 0.08, 0.26])
        # Normalize to sum to 1.0
        category_weights = category_weights / category_weights.sum()
        
        np.random.seed(42)
        
        for i in range(num_transactions):
            category = np.random.choice(categories, p=category_weights)
            
            if category == 'Income':
                # Adjusted income range: ₹15,000 to ₹25,000 per month => ₹18-30L annually, peaking at 20-25L as requested
                amount = np.random.uniform(15000, 25000)
            elif category == 'Groceries':
                # Weekly/monthly groceries: ₹300-1500
                amount = np.random.uniform(300, 1500)
            elif category == 'Transportation':
                # Fuel, cab, auto, metro: ₹100-800
                amount = np.random.uniform(100, 800)
            elif category == 'Entertainment':
                # Movies, subscriptions, outings: ₹200-1200
                amount = np.random.uniform(200, 1200)
            elif category == 'Utilities':
                # Electricity, water, internet, phone: ₹300-2000
                amount = np.random.uniform(300, 2000)
            elif category == 'Healthcare':
                # Doctor visits, medicines, tests: ₹150-2500
                amount = np.random.uniform(150, 2500)
            elif category == 'Education':
                # Tuition, books, courses: ₹500-8000
                amount = np.random.uniform(500, 8000)
            elif category == 'Dining Out':
                # Restaurants, food delivery: ₹200-1500
                amount = np.random.uniform(200, 1500)
            elif category == 'Clothing':
                # Clothes, shoes, accessories: ₹300-3000
                amount = np.random.uniform(300, 3000)
            elif category == 'Shopping':
                # General shopping, household items: ₹150-2000
                amount = np.random.uniform(150, 2000)
            else:
                amount = np.random.uniform(50, 1000)
            
            base_date = datetime.now() - timedelta(days=365)
            random_days = np.random.randint(0, 365)
            transaction_date = base_date + timedelta(days=random_days)
            
            self.add_transaction(
                amount=round(amount, 2),
                category=category,
                description=f"{category} - {self._generate_description(category)}",
                date=transaction_date.strftime('%Y-%m-%d')
            )

    def _generate_description(self, category):
        """Generate realistic transaction descriptions for Indian context"""
        descriptions = {
            'Groceries': ['Big Basket order', 'Local market shopping', 'Supermarket bill', 'Vegetables & fruits', 'Monthly groceries'],
            'Transportation': ['Petrol fill-up', 'Uber ride', 'Metro card recharge', 'Auto rickshaw', 'Cab to office'],
            'Entertainment': ['Movie tickets', 'Netflix subscription', 'Concert tickets', 'Gaming purchase', 'Book purchase'],
            'Utilities': ['Electricity bill', 'Internet bill', 'Mobile recharge', 'Water bill', 'Gas cylinder'],
            'Healthcare': ['Doctor consultation', 'Pharmacy medicines', 'Health checkup', 'Dental treatment', 'Eye test'],
            'Education': ['School fees', 'Online course', 'Books purchase', 'Exam fees', 'Educational materials'],
            'Dining Out': ['Restaurant bill', 'Zomato order', 'Swiggy delivery', 'Cafe visit', 'Street food'],
            'Clothing': ['Shirt purchase', 'Footwear', 'Online shopping', 'Ethnic wear', 'Winter clothes'],
            'Shopping': ['Amazon order', 'Electronics', 'Home decor', 'Kitchen items', 'Personal care'],
            'Income': ['Salary credit', 'Freelance payment', 'Bonus', 'Investment return', 'Other income']
        }
        
        return np.random.choice(descriptions.get(category, ['General expense']))

    def prepare_data(self):
        """Prepare data for machine learning models"""
        if not self.transactions:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.transactions)
        df['date'] = pd.to_datetime(df['date'])
        df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
        df['is_income'] = df['category'].apply(lambda x: 1 if x == 'Income' else 0)
        
        for col in ['category']:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col])
            else:
                try:
                    df[f'{col}_encoded'] = self.label_encoders[col].transform(df[col])
                except ValueError:
                    self.label_encoders[col] = LabelEncoder()
                    df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col])
        
        return df

    def train_models(self):
        """Train both ML models using Random Forest and Isolation Forest"""
        df = self.prepare_data()
        if len(df) < 10:
            return 0, 0
        
        try:
            features = ['day_of_week', 'month', 'is_weekend', 'category_encoded']
            X = df[features]
            y = df['amount']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            self.spending_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            self.spending_model.fit(X_train_scaled, y_train)
            
            predictions = self.spending_model.predict(X_test_scaled)
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            
            anomaly_features = ['amount', 'day_of_week', 'month', 'category_encoded']
            X_anomaly = df[anomaly_features]
            X_anomaly_scaled = self.scaler.fit_transform(X_anomaly)
            
            self.anomaly_detector = IsolationForest(
                contamination=0.1,
                random_state=42,
                n_estimators=100
            )
            self.anomaly_detector.fit(X_anomaly_scaled)
            
            return mae, r2
            
        except Exception as e:
            st.error(f"Error training models: {str(e)}")
            return 0, 0

    def detect_anomalies(self):
        """Detect unusual spending patterns using trained anomaly detector"""
        if self.anomaly_detector is None:
            return []
        
        df = self.prepare_data()
        if len(df) == 0:
            return []
        
        try:
            anomaly_features = ['amount', 'day_of_week', 'month', 'category_encoded']
            X_anomaly = df[anomaly_features]
            X_anomaly_scaled = self.scaler.transform(X_anomaly)
            
            anomaly_scores = self.anomaly_detector.decision_function(X_anomaly_scaled)
            anomalies = self.anomaly_detector.predict(X_anomaly_scaled) == -1
            
            anomaly_transactions = df[anomalies].copy()
            anomaly_transactions['anomaly_score'] = anomaly_scores[anomalies]
            
            return anomaly_transactions.sort_values(by='anomaly_score').head(10).to_dict('records')
            
        except Exception as e:
            return []

    def get_financial_insights(self):
        """Generate comprehensive AI-powered financial insights for Indian household"""
        df = self.prepare_data()
        insights = []
        
        if len(df) == 0:
            return ["Please generate sample data to view detailed financial insights and analytics."]
        
        try:
            expense_df = df[df['category'] != 'Income']
            income_df = df[df['category'] == 'Income']
            
            if len(expense_df) > 0:
                monthly_spending = expense_df.groupby('month')['amount'].sum()
                avg_monthly = monthly_spending.mean()
                insights.append(f"Your average monthly spending is ₹{avg_monthly:,.2f}. This provides a baseline for budget planning and financial forecasting in the Indian context.")
                
                category_spending = expense_df.groupby('category')['amount'].sum().sort_values(ascending=False)
                if len(category_spending) > 0:
                    top_category = category_spending.index[0]
                    top_amount = category_spending.iloc[0]
                    top_percentage = (top_amount / category_spending.sum()) * 100
                    insights.append(f"Your highest spending category is {top_category} at ₹{top_amount:,.2f} ({top_percentage:.1f}% of total expenses). Consider reviewing transactions in this category for optimization opportunities.")
                
                weekend_spending = expense_df[expense_df['is_weekend'] == 1]['amount'].mean()
                weekday_spending = expense_df[expense_df['is_weekend'] == 0]['amount'].mean()
                
                if pd.notna(weekend_spending) and pd.notna(weekday_spending) and weekday_spending > 0:
                    if weekend_spending > weekday_spending:
                        diff = ((weekend_spending/weekday_spending-1)*100)
                        insights.append(f"You spend {diff:.1f}% more on weekends (₹{weekend_spending:.2f} vs ₹{weekday_spending:.2f}). Weekend budgeting could help control discretionary spending.")
                    else:
                        diff = ((weekday_spending/weekend_spending-1)*100)
                        insights.append(f"Your weekday spending is {diff:.1f}% higher than weekends (₹{weekday_spending:.2f} vs ₹{weekend_spending:.2f}). This suggests disciplined weekend spending habits.")
            
            total_income = income_df['amount'].sum()
            total_expenses = expense_df['amount'].sum()
            
            if total_income > 0:
                savings_rate = ((total_income - total_expenses) / total_income) * 100
                monthly_savings = (total_income - total_expenses) / 12
                insights.append(f"Your current savings rate is {savings_rate:.1f}% with approximately ₹{monthly_savings:,.2f} saved per month. Financial experts recommend a minimum 20% savings rate for long-term financial health.")
            
            total_transactions = len(df)
            avg_transaction = df['amount'].mean()
            daily_transactions = total_transactions / 365
            insights.append(f"You average {daily_transactions:.1f} transactions per day with an average value of ₹{avg_transaction:.2f}. This indicates your spending frequency and typical transaction size patterns.")
            
            anomalies = self.detect_anomalies()
            if anomalies:
                insights.append(f"Our AI has identified {len(anomalies)} unusual spending patterns in your recent transactions. These outliers may represent opportunities for spending optimization or require attention.")
            
        except Exception as e:
            insights.append(f"Error generating insights: {str(e)}")
        
        return insights if insights else ["Generate sample data to unlock comprehensive financial insights and analytics."]

    def get_budgeting_recommendations(self):
        """Generate comprehensive AI-powered budgeting recommendations for Indian household"""
        df = self.prepare_data()
        recommendations = []
        
        if len(df) == 0:
            return ["Add transaction data to receive personalized budgeting recommendations tailored to your spending patterns."]
        
        try:
            expense_df = df[df['category'] != 'Income']
            if len(expense_df) == 0:
                return ["Add expense transactions to generate targeted budgeting recommendations and financial planning advice."]
            
            category_spending = expense_df.groupby('category')['amount'].agg(['sum', 'mean', 'count'])
            total_spending = category_spending['sum'].sum()
            
            if total_spending == 0:
                return ["Insufficient spending data available for comprehensive budgeting recommendations."]
            
            for category in category_spending.index:
                category_total = category_spending.loc[category, 'sum']
                percentage = (category_total / total_spending) * 100
                avg_transaction = category_spending.loc[category, 'mean']
                transaction_count = category_spending.loc[category, 'count']
                
                if percentage > 35:
                    recommendations.append(
                        f"PRIORITY: {category} represents {percentage:.1f}% of your expenses (₹{category_total:,.2f}). Consider implementing a strict budget limit and exploring cost-effective alternatives. This category significantly impacts your financial health."
                    )
                elif percentage > 25:
                    recommendations.append(
                        f"ATTENTION: {category} accounts for {percentage:.1f}% of spending with {transaction_count} transactions. Monitor this category closely and set monthly limits to prevent overspending."
                    )
                elif percentage > 15:
                    recommendations.append(
                        f"MONITOR: {category} spending is {percentage:.1f}% of your budget. This is within reasonable limits but track trends to ensure it doesn't increase disproportionately."
                    )
                
                if avg_transaction > 3000 and category not in ['Utilities', 'Healthcare', 'Income', 'Education']:
                    recommendations.append(
                        f"OPTIMIZE: Your average {category} transaction is ₹{avg_transaction:.2f}. Consider ways to reduce transaction sizes through bulk purchasing, discounts, or alternative providers."
                    )
                
                monthly_frequency = transaction_count / 12
                if monthly_frequency > 15 and category in ['Groceries', 'Dining Out']:
                    recommendations.append(
                        f"FREQUENCY ALERT: You have {monthly_frequency:.1f} {category} transactions per month. Consider meal planning or dining budgets to reduce frequency and costs."
                    )
            
            total_income = df[df['category'] == 'Income']['amount'].sum()
            if total_income > 0:
                savings_rate = ((total_income - total_spending) / total_income) * 100
                if savings_rate < 10:
                    shortfall = (total_income * 0.2) - (total_income - total_spending)
                    recommendations.append(f"CRITICAL: Your savings rate is {savings_rate:.1f}%, well below the recommended 20%. You need to save an additional ₹{shortfall:,.2f} annually to reach the minimum recommended savings rate.")
                elif savings_rate < 20:
                    target_savings = total_income * 0.2
                    current_savings = total_income - total_spending
                    additional_needed = target_savings - current_savings
                    recommendations.append(f"IMPROVE: Your savings rate is {savings_rate:.1f}%. To reach the recommended 20%, increase your monthly savings by ₹{additional_needed/12:.2f}.")
                elif savings_rate > 30:
                    excess_savings = (total_income - total_spending) - (total_income * 0.2)
                    recommendations.append(f"EXCELLENT: Your {savings_rate:.1f}% savings rate exceeds recommendations. Consider investing your excess ₹{excess_savings:,.2f} in mutual funds, SIPs, or FDs.")
            
            monthly_expenses = total_spending / 12
            emergency_fund_target = monthly_expenses * 6
            recommendations.append(f"EMERGENCY FUND: Based on your monthly expenses of ₹{monthly_expenses:,.2f}, aim for an emergency fund of ₹{emergency_fund_target:,.2f} (6 months of expenses).")
            
        except Exception as e:
            recommendations.append(f"Error generating recommendations: {str(e)}")
        
        return recommendations if recommendations else ["Your financial habits appear well-balanced. Continue monitoring your spending patterns and maintain your current budgeting discipline."]

def create_advanced_charts(advisor):
    """Create sophisticated professional charts without built-in titles"""
    df = advisor.prepare_data()
    if len(df) == 0:
        return None, None, None, None
    
    colors = {
        'primary': '#667eea',
        'secondary': '#764ba2',
        'success': '#48bb78',
        'warning': '#ed8936',
        'danger': '#f56565',
        'info': '#4299e1',
        'gradient': ['#667eea', '#764ba2', '#48bb78', '#ed8936']
    }
    
    try:
        expense_df = df[df['category'] != 'Income']
        
        # 1. Monthly Spending Trend
        monthly_data = expense_df.groupby('month')['amount'].agg(['sum', 'count', 'mean']).reset_index()
        if len(monthly_data) > 0:
            month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                          7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
            monthly_data['month_name'] = monthly_data['month'].map(month_names)
            
            fig_trend = go.Figure()
            
            fig_trend.add_trace(go.Scatter(
                x=monthly_data['month_name'],
                y=monthly_data['sum'],
                mode='lines+markers',
                name='Monthly Spending',
                line=dict(color=colors['primary'], width=4, shape='spline'),
                marker=dict(size=10, color=colors['primary'], line=dict(color='white', width=3)),
                fill='tonexty',
                fillcolor='rgba(102, 126, 234, 0.1)',
                hovertemplate='<b>%{x}</b><br>Total Spent: <b>₹%{y:,.2f}</b><br>Transactions: <b>%{customdata}</b><extra></extra>',
                customdata=monthly_data['count']
            ))
            
            avg_spending = monthly_data['sum'].mean()
            fig_trend.add_hline(
                y=avg_spending, 
                line_dash="dash", 
                line_color=colors['warning'],
                annotation_text=f"Average: ₹{avg_spending:,.2f}"
            )
            
            fig_trend.update_layout(
                xaxis_title='Month',
                yaxis_title='Amount Spent (₹)',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Source Sans Pro, sans-serif', size=11),
                hovermode='x unified',
                showlegend=False,
                margin=dict(t=20, b=50, l=50, r=50)
            )
        else:
            fig_trend = None
        
        # 2. Category Pie Chart
        if len(expense_df) > 0:
            category_data = expense_df.groupby('category').agg({
                'amount': ['sum', 'count', 'mean']
            }).round(2)
            category_data.columns = ['total', 'count', 'average']
            category_data = category_data.reset_index()
            
            fig_pie = go.Figure(data=[go.Pie(
                labels=category_data['category'],
                values=category_data['total'],
                hole=0.4,
                hovertemplate='<b>%{label}</b><br>Total: <b>₹%{value:,.2f}</b><br>Percentage: <b>%{percent}</b><br>Transactions: <b>%{customdata}</b><extra></extra>',
                customdata=category_data['count'],
                textinfo='label+percent',
                textposition='outside',
                textfont=dict(size=10),
                marker=dict(
                    colors=px.colors.qualitative.Set3,
                    line=dict(color='white', width=2)
                )
            )])
            
            fig_pie.update_layout(
                font=dict(family='Source Sans Pro, sans-serif', size=11),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(t=20, b=50, l=50, r=50)
            )
        else:
            fig_pie = None
        
        # 3. Weekly Pattern
        if len(expense_df) > 0:
            weekly_data = expense_df.groupby('day_of_week').agg({
                'amount': ['mean', 'std', 'count']
            }).round(2)
            weekly_data.columns = ['mean', 'std', 'count']
            weekly_data = weekly_data.reset_index()
            
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            weekly_data['day_name'] = weekly_data['day_of_week'].map(
                {i: day_names[i] for i in range(7)}
            )
            
            fig_weekly = go.Figure()
            
            fig_weekly.add_trace(go.Bar(
                x=weekly_data['day_name'],
                y=weekly_data['mean'],
                error_y=dict(type='data', array=weekly_data['std'], visible=True),
                marker_color=colors['primary'],
                opacity=0.8,
                hovertemplate='<b>%{x}</b><br>Average: <b>₹%{y:.2f}</b><br>Std Dev: <b>₹%{customdata:.2f}</b><extra></extra>',
                customdata=weekly_data['std']
            ))
            
            fig_weekly.update_layout(
                xaxis_title='Day of Week',
                yaxis_title='Average Amount (₹)',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Source Sans Pro, sans-serif', size=11),
                showlegend=False,
                margin=dict(t=20, b=50, l=50, r=50)
            )
        else:
            fig_weekly = None
        
        # 4. Income vs Expenses
        if len(df) > 0:
            monthly_summary = df.groupby(['month', 'category']).agg({
                'amount': 'sum'
            }).reset_index()
            
            fig_income_expense = go.Figure()
            
            month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                          7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
            
            income_data = monthly_summary[monthly_summary['category'] == 'Income']
            if len(income_data) > 0:
                fig_income_expense.add_trace(go.Bar(
                    x=[month_names.get(m, str(m)) for m in income_data['month']],
                    y=income_data['amount'],
                    name='Income',
                    marker_color=colors['success'],
                    opacity=0.8
                ))
            
            expense_summary = monthly_summary[monthly_summary['category'] != 'Income'].groupby('month')['amount'].sum().reset_index()
            if len(expense_summary) > 0:
                fig_income_expense.add_trace(go.Bar(
                    x=[month_names.get(m, str(m)) for m in expense_summary['month']],
                    y=expense_summary['amount'],
                    name='Expenses',
                    marker_color=colors['danger'],
                    opacity=0.8
                ))
            
            fig_income_expense.update_layout(
                xaxis_title='Month',
                yaxis_title='Amount (₹)',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Source Sans Pro, sans-serif', size=11),
                barmode='group',
                hovermode='x unified',
                margin=dict(t=20, b=50, l=50, r=50)
            )
        else:
            fig_income_expense = None
        
        return fig_trend, fig_pie, fig_weekly, fig_income_expense
        
    except Exception as e:
        st.error(f"Error creating charts: {str(e)}")
        return None, None, None, None

def render_enhanced_metrics(advisor, view_period):
    """Render enhanced metrics dashboard with INR currency and period selection"""
    df = advisor.prepare_data()
    if len(df) == 0:
        return
    
    total_income = df[df['category'] == 'Income']['amount'].sum()
    total_expenses = df[df['category'] != 'Income']['amount'].sum()
    balance = total_income - total_expenses
    
    # Calculate metrics based on selected period
    if view_period == "Monthly":
        period_divisor = 12
        period_text = "Monthly"
        avg_income = total_income / period_divisor
        avg_expenses = total_expenses / period_divisor
        avg_balance = balance / period_divisor
    else:  # Annual
        period_divisor = 1
        period_text = "Annual"
        avg_income = total_income
        avg_expenses = total_expenses
        avg_balance = balance
    
    savings_rate = ((total_income - total_expenses) / total_income * 100) if total_income > 0 else 0
    savings_delta = savings_rate - 20
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">{period_text} Balance</div>
            <div class="metric-value">₹{avg_balance:,.2f}</div>
            <div class="metric-delta" style="color: {'#48bb78' if avg_balance > 0 else '#f56565'}">
                {'Positive' if avg_balance > 0 else 'Negative'} balance
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">{period_text} Income</div>
            <div class="metric-value">₹{avg_income:,.2f}</div>
            <div class="metric-delta" style="color: #4299e1">
                {'Total yearly' if view_period == 'Annual' else 'Average per month'}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">{period_text} Expenses</div>
            <div class="metric-value">₹{avg_expenses:,.2f}</div>
            <div class="metric-delta" style="color: #ed8936">
                {'Total yearly' if view_period == 'Annual' else 'Average per month'}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Savings Rate</div>
            <div class="metric-value">{savings_rate:.1f}%</div>
            <div class="metric-delta" style="color: {'#48bb78' if savings_delta > 0 else '#f56565'}">
                {'+' if savings_delta > 0 else ''}{savings_delta:.1f}% vs recommended
            </div>
        </div>
        """, unsafe_allow_html=True)

def render_transaction_form(advisor):
    """Render professional transaction input form without emojis"""
    with st.expander("Add New Transaction", expanded=False):
        with st.form("transaction_form", clear_on_submit=True):
            st.markdown('<div class="form-header">Enter Transaction Details</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                amount = st.number_input(
                    "Amount (₹)", 
                    min_value=0.01, 
                    step=0.01,
                    value=100.00,
                    help="Enter the transaction amount in Indian Rupees"
                )
                category = st.selectbox(
                    "Category",
                    options=['Groceries', 'Transportation', 'Entertainment', 'Utilities', 
                            'Healthcare', 'Education', 'Dining Out', 'Clothing', 'Shopping'],
                    help="Select the appropriate expense category"
                )
            
            with col2:
                transaction_date = st.date_input(
                    "Transaction Date",
                    value=datetime.now(),
                    help="Select when the transaction occurred"
                )
                transaction_type = st.radio(
                    "Transaction Type",
                    options=['Expense', 'Income'],
                    horizontal=True,
                    help="Choose whether this is income or an expense"
                )
            
            description = st.text_area(
                "Description (Optional)", 
                height=80,
                placeholder="Enter additional details about this transaction...",
                help="Add any notes or details about this transaction"
            )
            
            if st.form_submit_button("Add Transaction", type="primary", use_container_width=True):
                if transaction_type == 'Expense':
                    final_category = category
                else:
                    final_category = 'Income'
                    
                advisor.add_transaction(
                    amount=amount,
                    category=final_category,
                    description=description or f"{final_category} - {amount:.2f}",
                    date=transaction_date.strftime('%Y-%m-%d')
                )
                st.success(f"Transaction added successfully! ₹{amount:,.2f} recorded as {final_category}")
                st.rerun()

def export_data_functionality(advisor):
    """Enhanced data export functionality with professional formatting"""
    df = advisor.prepare_data()
    if len(df) == 0:
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # CSV Export
        export_df = df[['date', 'category', 'amount', 'description']].copy()
        export_df['amount'] = export_df['amount'].apply(lambda x: f"₹{x:,.2f}")
        csv = export_df.to_csv(index=False)
        st.download_button(
            label="Download CSV Report",
            data=csv,
            file_name=f"FinIQ_Financial_Report_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        # Excel Export with multiple sheets
        excel_buffer = io.BytesIO()
        
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            # Transaction Details sheet
            transaction_df = df[['date', 'category', 'amount', 'description']].copy()
            transaction_df.to_excel(writer, sheet_name='Transaction_Details', index=False)
            
            # Monthly Summary sheet
            expense_df = df[df['category'] != 'Income']
            monthly_summary = expense_df.groupby(['month', 'category'])['amount'].sum().unstack(fill_value=0)
            monthly_summary.to_excel(writer, sheet_name='Monthly_Summary_by_Category')
            
            # Category Analysis sheet
            category_analysis = expense_df.groupby('category')['amount'].agg([
                ('Total_Spent', 'sum'),
                ('Average_Transaction', 'mean'),
                ('Transaction_Count', 'count'),
                ('Minimum_Amount', 'min'),
                ('Maximum_Amount', 'max')
            ]).round(2)
            category_analysis.to_excel(writer, sheet_name='Category_Analysis')
            
            # Financial Overview sheet
            total_income = df[df['category'] == 'Income']['amount'].sum()
            total_expenses = expense_df['amount'].sum()
            savings_rate = ((total_income - total_expenses) / total_income * 100) if total_income > 0 else 0
            
            overview_data = {
                'Financial_Metric': [
                    'Total Income', 'Total Expenses', 'Net Savings', 'Savings Rate (%)',
                    'Average Monthly Income', 'Average Monthly Expenses'
                ],
                'Amount_INR': [
                    total_income, total_expenses, total_income - total_expenses, savings_rate,
                    total_income / 12, total_expenses / 12
                ]
            }
            overview_df = pd.DataFrame(overview_data)
            overview_df.to_excel(writer, sheet_name='Financial_Overview', index=False)
        
        st.download_button(
            label="Download Excel Report",
            data=excel_buffer.getvalue(),
            file_name=f"FinIQ_Comprehensive_Report_{datetime.now().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    
    with col3:
        # JSON Export
        json_data = df.to_json(orient='records', date_format='iso')
        st.download_button(
            label="Download JSON Data",
            data=json_data,
            file_name=f"FinIQ_Data_Export_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json",
            use_container_width=True
        )

# Initialize session state and advisor
if 'advisor' not in st.session_state:
    st.session_state.advisor = StreamlitFinanceAdvisor()
    st.session_state.data_generated = False
    st.session_state.models_trained = False

# Initialize view period if not exists
if 'view_period' not in st.session_state:
    st.session_state.view_period = 'Monthly'

# OPTIMIZED: Initialize placeholders for non-blocking messages
if 'data_placeholder' not in st.session_state:
    st.session_state.data_placeholder = None

if 'train_placeholder' not in st.session_state:
    st.session_state.train_placeholder = None

if 'anomaly_alert_time' not in st.session_state:
    st.session_state.anomaly_alert_time = None

advisor = st.session_state.advisor

# Enhanced Professional Sidebar with Proper Diamond Icon
with st.sidebar:
    st.markdown('''
    <div class="sidebar-title">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M12 2L19 12L12 22L5 12L12 2Z" fill="#667eea" stroke="#764ba2" stroke-width="1"/>
        </svg>
        FinIQ
    </div>
    ''', unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-section-header">Data Management</div>', unsafe_allow_html=True)
    
    # Create placeholders for optimized button responses
    data_placeholder = st.empty()
    train_placeholder = st.empty()
    
    # OPTIMIZED: Generate Sample Data button with instant feedback
    if st.button("Generate Sample Data", use_container_width=True):
        advisor.transactions.clear()
        st.session_state.data_generated = False
        st.session_state.models_trained = False
        
        with data_placeholder:
            with st.spinner("Generating fresh Indian household data..."):
                advisor.generate_sample_data(1000)
                st.session_state.data_generated = True
        
        data_placeholder.success("Fresh sample data generated successfully!")
        st.session_state.data_success_time = time.time()
        st.rerun()
    
    # Auto-hide data generation success message
    if 'data_success_time' in st.session_state:
        if time.time() - st.session_state.data_success_time > 3:
            data_placeholder.empty()
            del st.session_state.data_success_time
    
    # OPTIMIZED: Train AI Models with instant feedback
    if st.button("Train AI Models", use_container_width=True):
        if st.session_state.data_generated:
            with train_placeholder:
                with st.spinner("Training advanced AI models..."):
                    mae, r2 = advisor.train_models()
                    st.session_state.models_trained = True
                    st.session_state.train_results = (mae, r2)
            
            train_placeholder.success(f"Models trained successfully!\n\nPrediction Error: ₹{mae:.2f}\nAccuracy Score: {r2:.3f}")
            st.session_state.train_success_time = time.time()
        else:
            train_placeholder.warning("Please generate data first")
    
    # Auto-hide training success message
    if 'train_success_time' in st.session_state:
        if time.time() - st.session_state.train_success_time > 4:
            train_placeholder.empty()
            del st.session_state.train_success_time
    
    if st.button("Refresh Dashboard", use_container_width=True):
        st.rerun()
    
    if st.session_state.data_generated:
        st.markdown('<div class="sidebar-section-header">Filters & Analysis</div>', unsafe_allow_html=True)
        
        # Period Selection Filter
        current_period = st.session_state.get('view_period', 'Monthly')
        period_index = ['Monthly', 'Annual'].index(current_period)
        
        view_period = st.selectbox(
            "View Metrics By:",
            options=["Monthly", "Annual"],
            index=period_index,
            help="Choose how to display financial metrics"
        )
        
        # Update session state when selection changes
        st.session_state.view_period = view_period
        
        df = advisor.prepare_data()
        if len(df) > 0:
            show_anomalies = st.checkbox("Show Anomaly Alerts", value=True)
            # OPTIMIZED: Anomaly alert with session-state controlled timing
            if show_anomalies and st.session_state.models_trained:
                anomalies = advisor.detect_anomalies()
                if anomalies and st.session_state.anomaly_alert_time is None:
                    st.warning(f"⚠️ {len(anomalies)} unusual transactions detected")
                    st.session_state.anomaly_alert_time = time.time()

# Main Content Area
st.markdown("""
<div class='main-header'>
    <h1>FinIQ</h1>
    <p>Advanced Financial Intelligence & Analytics Platform</p>
</div>
""", unsafe_allow_html=True)

if not st.session_state.data_generated:
    st.info("Generate sample data from the sidebar to begin your comprehensive financial analysis.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="welcome-section">
        <h3>Advanced Analytics</h3>
        <ul>
            <li><strong>Machine Learning Models:</strong> Random Forest prediction and Isolation Forest anomaly detection</li>
            <li><strong>Pattern Recognition:</strong> Identify spending trends and behavioral patterns</li>
            <li><strong>Predictive Insights:</strong> Forecast future spending based on historical data</li>
            <li><strong>Statistical Analysis:</strong> Comprehensive evaluation of financial habits</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="welcome-section">
        <h3>Smart Management</h3>
        <ul>
            <li><strong>Automated Categorization:</strong> AI-powered transaction classification</li>
            <li><strong>Budget Optimization:</strong> Personalized recommendations for Indian households</li>
            <li><strong>Anomaly Detection:</strong> Identify unusual transactions and spending patterns</li>
            <li><strong>Goal Tracking:</strong> Monitor progress towards financial objectives</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="welcome-section">
        <h3>Professional Reporting</h3>
        <ul>
            <li><strong>Interactive Visualizations:</strong> Advanced charts with drill-down capabilities</li>
            <li><strong>Comprehensive Dashboards:</strong> Multi-dimensional financial analysis</li>
            <li><strong>Export Capabilities:</strong> Professional Excel, CSV, and JSON reports</li>
            <li><strong>Indian Context:</strong> Tailored for upper middle class household patterns</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

else:
    # Enhanced metrics dashboard with working period selection
    current_view_period = st.session_state.get('view_period', 'Monthly')
    render_enhanced_metrics(advisor, current_view_period)
    
    # Professional transaction input form
    render_transaction_form(advisor)
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Advanced Analytics", "Financial Insights", "Smart Recommendations", "Unusual Transactions", "Transaction Management"])
    
    with tab1:
        st.markdown('<div class="section-header">Advanced Financial Analytics Dashboard</div>', unsafe_allow_html=True)
        
        charts = create_advanced_charts(advisor)
        
        if charts and charts[0] is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                if charts[0]:
                    st.plotly_chart(charts[0], use_container_width=True)
                    st.markdown('<div class="chart-title">Monthly Spending Analysis with Trend</div>', unsafe_allow_html=True)
                    
            with col2:
                if charts[1]:
                    st.plotly_chart(charts[1], use_container_width=True)
                    st.markdown('<div class="chart-title">Spending Distribution by Category</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if charts[2]:
                    st.plotly_chart(charts[2], use_container_width=True)
                    st.markdown('<div class="chart-title">Daily Spending Patterns with Variability</div>', unsafe_allow_html=True)
            
            with col2:
                if charts[3]:
                    st.plotly_chart(charts[3], use_container_width=True)
                    st.markdown('<div class="chart-title">Monthly Income vs Expenses Comparison</div>', unsafe_allow_html=True)
                    
        else:
            st.info("Interactive charts will appear here once you have sufficient transaction data for analysis.")
    
    with tab2:
        st.markdown('<div class="section-header">AI-Powered Financial Insights</div>', unsafe_allow_html=True)
        insights = advisor.get_financial_insights()
        
        for i, insight in enumerate(insights, 1):
            st.markdown(f"""
            <div class='insight-card'>
                <h4>Financial Insight {i}</h4>
                <p>{insight}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with tab3:
        st.markdown('<div class="section-header">Personalized Financial Recommendations</div>', unsafe_allow_html=True)
        recommendations = advisor.get_budgeting_recommendations()
        
        for i, rec in enumerate(recommendations, 1):
            st.markdown(f"""
            <div class='recommendation-card'>
                <h4>Recommendation {i}</h4>
                <p>{rec}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with tab4:
        st.markdown('<div class="section-header">Unusual Transactions Detection</div>', unsafe_allow_html=True)
        
        if st.session_state.models_trained:
            anomalies = advisor.detect_anomalies()
            
            if anomalies:
                # FIXED: Blue info dialog with proper state management
                info_placeholder = st.empty()
                
                # Show info only once and manage timing via session state
                if 'anomaly_info_shown' not in st.session_state:
                    st.session_state.anomaly_info_shown = True
                    st.session_state.anomaly_info_time = time.time()
                
                # Display info if within time limit
                if time.time() - st.session_state.get('anomaly_info_time', 0) < 4:
                    info_placeholder.info(f"Our AI has detected {len(anomalies)} unusual transactions based on your spending patterns. These transactions deviate significantly from your normal behavior and may require attention.")
                else:
                    info_placeholder.empty()
                
                for i, anomaly in enumerate(anomalies, 1):
                    date_formatted = pd.to_datetime(anomaly['date']).strftime('%d %B %Y')
                    
                    # Determine severity based on anomaly score
                    if anomaly['anomaly_score'] < -0.3:
                        severity = "HIGH PRIORITY"
                        severity_color = "#c53030"
                    elif anomaly['anomaly_score'] < -0.1:
                        severity = "MEDIUM PRIORITY"
                        severity_color = "#dd6b20"
                    else:
                        severity = "LOW PRIORITY"
                        severity_color = "#38a169"
                    
                    st.markdown(f"""
                    <div class='anomaly-card'>
                        <h4>Unusual Transaction #{i} - <span style="color: {severity_color}">{severity}</span></h4>
                        <p><strong>Date:</strong> {date_formatted}</p>
                        <p><strong>Category:</strong> {anomaly['category']}</p>
                        <p><strong>Amount:</strong> ₹{anomaly['amount']:,.2f}</p>
                        <p><strong>Description:</strong> {anomaly['description']}</p>
                        <p><strong>Anomaly Score:</strong> {anomaly['anomaly_score']:.3f} (lower scores indicate higher unusual activity)</p>
                        <p><strong>Analysis:</strong> This transaction is unusual because it significantly deviates from your typical {anomaly['category']} spending pattern in terms of amount and timing.</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.success("No unusual transactions detected! Your spending patterns appear consistent and normal.")
                st.info("Our AI Anomaly Detection system continuously monitors your transactions for unusual patterns. Anomalies might include unusually large expenses, purchases in uncommon categories, or spending at unusual times.")
        else:
            st.warning("Please train the AI models first to enable unusual transaction detection.")
            st.info("The Anomaly Detection feature uses Isolation Forest machine learning to identify transactions that significantly deviate from your normal spending patterns.")
    
    with tab5:
        st.markdown('<div class="section-header">Transaction Management System</div>', unsafe_allow_html=True)
        
        df = advisor.prepare_data()
        if len(df) > 0:
            display_df = df[['date', 'category', 'amount', 'description']].sort_values('date', ascending=False)
            
            st.dataframe(
                display_df,
                column_config={
                    "amount": st.column_config.NumberColumn(
                        "Amount (₹)",
                        format="₹%.2f",
                        help="Transaction amount in Indian Rupees"
                    ),
                    "date": st.column_config.DateColumn(
                        "Date",
                        help="Transaction date"
                    ),
                    "category": st.column_config.TextColumn(
                        "Category",
                        help="Transaction category"
                    ),
                    "description": st.column_config.TextColumn(
                        "Description",
                        help="Transaction description"
                    )
                },
                use_container_width=True,
                hide_index=True,
                height=400
            )
            
            st.markdown("### Professional Export Options")
            export_data_functionality(advisor)
            
        else:
            st.info("No transactions available. Generate sample data to see transaction management features.")

# Footer with technical information
st.markdown("---")
st.markdown("""
**Technical Implementation:** This application uses Random Forest Regression for spending prediction and Isolation Forest for anomaly detection. 
The sample data represents typical upper middle class Indian household spending patterns with amounts in Indian Rupees (₹). 
Export functionality provides professionally formatted Excel sheets with multiple tabs for comprehensive financial analysis.
""")
