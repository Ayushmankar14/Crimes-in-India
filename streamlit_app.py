import streamlit as st
import pandas as pd
import numpy as np
<<<<<<< HEAD
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="📱 Google Play ML App", layout="wide")

# -------------------- DATA CLEANING ------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("googleplaystore.csv")
    df = df[df['Rating'].notnull() & (df['Rating'] <= 5)]

    df['Installs'] = pd.to_numeric(df['Installs'].str.replace('[+,]', '', regex=True), errors='coerce')
    df['Installs'].fillna(df['Installs'].median(), inplace=True)

    df['Price'] = pd.to_numeric(df['Price'].str.replace('$', '', regex=True), errors='coerce')
    df['Price'] = df['Price'].fillna(0)

    df['Reviews'] = pd.to_numeric(df['Reviews'], errors='coerce')
    df['Reviews'] = df['Reviews'].fillna(df['Reviews'].median() if not df['Reviews'].dropna().empty else 1000)

    def clean_size(val):
        if 'M' in val:
            return float(val.replace('M', '')) * 1_000_000
        elif 'k' in val:
            return float(val.replace('k', '')) * 1_000
        else:
            return np.nan

    df['Size'] = df['Size'].astype(str).apply(clean_size)
    df['Size'].fillna(df['Size'].median(), inplace=True)

    df.drop(['App', 'Last Updated', 'Current Ver', 'Android Ver'], axis=1, inplace=True)

    le = LabelEncoder()
    for col in ['Category', 'Content Rating', 'Genres', 'Type']:
        df[col] = le.fit_transform(df[col].astype(str))

    return df

# -------------------- TRAINING & EVALUATION ------------------------
def train_models(X_train, y_train):
    models = {
        "Random Forest": RandomForestRegressor(),
        "Linear Regression": LinearRegression(),
        "Gradient Boosting": GradientBoostingRegressor()
    }
    trained = {name: model.fit(X_train, y_train) for name, model in models.items()}
    return trained

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    return {
        "MSE": mean_squared_error(y_test, preds),
        "MAE": mean_absolute_error(y_test, preds),
        "R²": r2_score(y_test, preds)
    }

# -------------------- DASHBOARD ------------------------
def page_dashboard(df):
    st.markdown("""
        <div style="background-color:#e6f2ff;padding:20px;border-radius:10px;margin-bottom:20px">
            <h1 style="color:#1e40af;text-align:center;">📱 Google Play Store ML Dashboard</h1>
        </div>
    """, unsafe_allow_html=True)

    st.sidebar.header("🎛️ Controls")
    target_choice = st.sidebar.radio("Prediction Target", ["Rating", "Installs"])
    model_choice = st.sidebar.selectbox("Model", ["Random Forest", "Linear Regression", "Gradient Boosting"])

    st.sidebar.subheader("Input Features")
    input_data = {
        "Category": st.sidebar.slider("Category", 0, int(df['Category'].max()), 5),
        "Reviews": st.sidebar.number_input("Reviews", value=5000),
        "Size": st.sidebar.number_input("Size (bytes)", value=10_000_000),
        "Installs": st.sidebar.number_input("Installs", value=100_000),
        "Type": st.sidebar.selectbox("Type", ['Free', 'Paid']),
        "Price": st.sidebar.number_input("Price", value=0.0),
        "Content Rating": st.sidebar.slider("Content Rating", 0, int(df['Content Rating'].max()), 3),
        "Genres": st.sidebar.slider("Genres", 0, int(df['Genres'].max()), 10)
    }

    X = df[['Category', 'Reviews', 'Size', 'Installs', 'Type', 'Price', 'Content Rating', 'Genres']]
    y = df[target_choice]
    X['Type'] = X['Type'].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = train_models(X_train, y_train)
    selected_model = models[model_choice]
    metrics = evaluate_model(selected_model, X_test, y_test)

    input_df = pd.DataFrame([[input_data['Category'], input_data['Reviews'], input_data['Size'],
                              input_data['Installs'], 0 if input_data['Type'] == 'Free' else 1,
                              input_data['Price'], input_data['Content Rating'], input_data['Genres']]],
                            columns=X.columns)

    prediction = selected_model.predict(input_df)[0]
    st.success(f"🎯 Predicted {target_choice}: {prediction:.2f}")
    st.json(metrics)

    st.markdown("### 📊 Visualizations")
    viz_option = st.selectbox("Choose plot", ["Rating Distribution", "Category vs Rating", "Installs vs Rating"])

    plt.clf()
    if viz_option == "Rating Distribution":
        fig = plt.figure(figsize=(6, 4))
        sns.histplot(df['Rating'], bins=20, kde=True, color='skyblue')
        st.pyplot(fig)
    elif viz_option == "Category vs Rating":
        fig = plt.figure(figsize=(8, 4))
        avg_rating = df.groupby('Category')['Rating'].mean().sort_values(ascending=False)
        sns.barplot(x=avg_rating.index, y=avg_rating.values, palette="viridis")
        plt.xticks(rotation=90)
        st.pyplot(fig)
    elif viz_option == "Installs vs Rating":
        fig = plt.figure(figsize=(7, 4))
        sns.scatterplot(data=df, x='Installs', y='Rating', hue='Category', alpha=0.6)
        st.pyplot(fig)

# -------------------- TOP APPS ------------------------
def page_top_apps(df):
    st.title("🔥 Top Apps Overview")
    search_term = st.text_input("🔍 Search by Category")
    if search_term:
        filtered = df[df['Category'].astype(str).str.contains(search_term, case=False)]
        st.dataframe(filtered[['Category', 'Rating', 'Reviews', 'Installs']].head(10))

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("⭐ Top Rated Apps")
        top_rated = df.sort_values(by='Rating', ascending=False).head(10)
        st.dataframe(top_rated[['Category', 'Rating', 'Reviews', 'Installs']])
    with col2:
        st.subheader("📈 Trending Apps")
        trending = df.sort_values(by='Installs', ascending=False).head(10)
        st.dataframe(trending[['Category', 'Rating', 'Reviews', 'Installs']])

# -------------------- HOME ------------------------
def page_home():
    st.markdown("""
        <div style="background-color:#e6f2ff;padding:30px;border-radius:10px;margin-bottom:20px">
            <h1 style="text-align:center; color:green;">🏠 Welcome to Google Play ML App</h1>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<p style="color:blue; font-size:18px;">Use the sidebar to navigate and explore features.</p>', unsafe_allow_html=True)

# -------------------- MAIN ------------------------
df = load_data()
page = st.sidebar.radio("📂 Pages", ["Home", "ML Dashboard", "Top Apps"])
if page == "Home":
    page_home()
elif page == "ML Dashboard":
    page_dashboard(df)
elif page == "Top Apps":
    page_top_apps(df)
=======
import pickle
import plotly.express as px

# ------------------ CONFIG ------------------
st.set_page_config(page_title="India Crime Dashboard", layout="wide")
st.markdown("""
<style>
footer {visibility: hidden;}
.stButton>button {background-color: #FF4B4B; color:white;}
.css-18e3th9 {padding: 1rem}
</style>
""", unsafe_allow_html=True)

# ------------------ LOGIN SYSTEM ------------------
USERS = {"admin": "1234", "guest": "pass"}

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login():
    st.title("🔐 Login to Crime Dashboard")
    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")
    if st.button("Login"):
        if user in USERS and USERS[user] == pwd:
            st.session_state.logged_in = True
            st.session_state.username = user
            st.success("✅ Login successful!")
            st.rerun()
        else:
            st.error("❌ Invalid credentials")

if not st.session_state.logged_in:
    login()
    st.stop()

# ------------------ LOGOUT ------------------
st.sidebar.title(f"Welcome, {st.session_state.username}")
if st.sidebar.button("🚪 Logout"):
    st.session_state.logged_in = False
    st.rerun()

# ------------------ LOAD MODELS ------------------
with open("models.pickle", "rb") as f:
    models = pickle.load(f)

with open("label_encoder.pickle", "rb") as f:
    label_encoder = pickle.load(f)

# ------------------ LOAD DATASETS ------------------
def load_csv(name):
    try:
        return pd.read_csv(name)
    except:
        return pd.DataFrame()

datasets = {
    "Cyber Crime": load_csv("cyber_crime.csv"),
    "Kidnapping": load_csv("kidnapping.csv"),
    "Murder Motives": load_csv("murder_motives.csv"),
    "Rape Victims": load_csv("rape_victims.csv"),
    "Crime Against Children": load_csv("crime_against_children.csv"),
    "Juvenile Crime": load_csv("juvenile_crime.csv"),
    "Missing Children": load_csv("missing_traced_children.csv"),
    "Trafficking": load_csv("trafficing.csv"),
}

# ------------------ ML PREDICTOR ------------------
st.title("🧠 Crime in India - ML Predictor & Dashboard")

with st.expander("🔮 Predict Crimes in 2016"):
    states = label_encoder.classes_
    state = st.selectbox("📍 Select State", states)
    state_encoded = label_encoder.transform([state])[0]

    col1, col2, col3 = st.columns(3)
    crime_2014 = col1.number_input("Crime in 2014", 0, value=10000)
    crime_2015 = col2.number_input("Crime in 2015", 0, value=10000)
    crime_rate = col3.number_input("Crime Rate", 0.0, value=200.0)
    crime_share = st.number_input("Crime Share % (State)", 0.0, value=3.0)
    model_choice = st.selectbox("🤖 Select ML Model", list(models.keys()))

    if st.button("Predict Now"):
        input_data = np.array([[state_encoded, crime_2014, crime_2015, crime_share, crime_rate]])
        prediction = models[model_choice].predict(input_data)[0]
        st.success(f"🔍 Predicted crimes in 2016 for **{state}**: **{int(prediction):,}**")

st.markdown("---")

# ------------------ PLOT HELPERS ------------------
def plot_bar(df, x, y, title):
    fig = px.bar(df, x=x, y=y, title=title, color=y, height=350)
    fig.update_layout(margin=dict(t=40, l=20, r=20, b=20))
    return fig

def plot_pie(series, title):
    fig = px.pie(names=series.index, values=series.values, title=title, hole=0.4)
    return fig

# ------------------ DASHBOARD ------------------
st.header("📊 Crime Insights Dashboard")

# --- Row 1 ---
col1, col2 = st.columns(2)
with col1:
    cyber = datasets["Cyber Crime"]
    if "CYBER CRIME (2016)" in cyber.columns:
        filtered = cyber[cyber["STATE NAME"] == state]
        if not filtered.empty:
            st.plotly_chart(plot_bar(filtered, "STATE NAME", "CYBER CRIME (2016)", f"Cyber Crime in {state} (2016)"), use_container_width=True)

with col2:
    kid = datasets["Kidnapping"]
    gender_cols = ["KIDNAPPED & ABDUCTED (MALE)", "KIDNAPPED & ABDUCTED (FEMALE)"]
    state_col = "STATE/UT" if "STATE/UT" in kid.columns else "STATE NAME"
    if all(col in kid.columns for col in gender_cols):
        filtered = kid[kid[state_col] == state]
        if not filtered.empty:
            totals = filtered[gender_cols].sum()
            st.plotly_chart(plot_pie(totals, f"Kidnapping Gender Split in {state}"), use_container_width=True)

# --- Row 2 ---
col3, col4 = st.columns(2)
with col3:
    motives = datasets["Murder Motives"]
    cols = ["PERSONAL VENDETTA OR ENMITY", "PROPERTY DISPUTE", "GAIN"]
    state_col = "STATE/UT" if "STATE/UT" in motives.columns else "STATE NAME"
    if all(c in motives.columns for c in cols):
        filtered = motives[motives[state_col] == state]
        if not filtered.empty:
            motive_sum = filtered[cols].sum()
            st.plotly_chart(plot_pie(motive_sum, f"Murder Motives in {state}"), use_container_width=True)

with col4:
    rape = datasets["Rape Victims"]
    if not rape.empty:
        age_ranges = [
            "RAPE VICTIMS BELOW 6 YEARS",
            "RAPE VICTIMS 6-11 YEARS",
            "RAPE VICTIMS 12-15 YEARS",
            "RAPE VICTIMS 16-17 YEARS",
            "RAPE VICTIMS 18-29 YEARS",
            "RAPE VICTIMS 30-44 YEARS",
            "RAPE VICTIMS 45-59 YEARS",
            "RAPE VICTIMS 60 YEARS & ABOVE"
        ]
        if all(col in rape.columns for col in age_ranges):
            melted = pd.melt(rape, id_vars="STATE NAME", value_vars=age_ranges, var_name="Age Group", value_name="Victim Count")
            filtered = melted[melted["STATE NAME"] == state]
            st.plotly_chart(px.bar(filtered, x="Age Group", y="Victim Count", color="Age Group", title=f"Rape Victims by Age Group in {state} (2016)"), use_container_width=True)

# --- Row 3 ---
col5, col6 = st.columns(2)
with col5:
    child = datasets["Crime Against Children"]
    if "STATE/UT" in child.columns and "TOTAL CRIME HEADS" in child.columns:
        filtered = child[child["STATE/UT"] == state]
        if not filtered.empty:
            st.plotly_chart(plot_bar(filtered, "STATE/UT", "TOTAL CRIME HEADS", f"Crimes Against Children in {state}"), use_container_width=True)

with col6:
    juv = datasets["Juvenile Crime"]
    if "STATE/UT" in juv.columns and "TOTAL" in juv.columns:
        filtered = juv[juv["STATE/UT"] == state]
        if not filtered.empty:
            jv = filtered.groupby("STATE/UT")["TOTAL"].sum().reset_index()
            st.plotly_chart(plot_bar(jv, "STATE/UT", "TOTAL", f"Juvenile Crimes in {state}"), use_container_width=True)

# --- Row 4 ---
col7, col8 = st.columns(2)
with col7:
    missing = datasets["Missing Children"]
    if "STATE/UT" in missing.columns and "TOTAL MISSING CHILDREN" in missing.columns:
        filtered = missing[missing["STATE/UT"] == state]
        if not filtered.empty:
            st.plotly_chart(plot_bar(filtered, "STATE/UT", "TOTAL MISSING CHILDREN", f"Missing Children in {state}"), use_container_width=True)

with col8:
    traf = datasets["Trafficking"]
    if "STATE/UT" in traf.columns and "TOTAL TRAFFICKED" in traf.columns:
        filtered = traf[traf["STATE/UT"] == state]
        if not filtered.empty:
            st.plotly_chart(plot_bar(filtered, "STATE/UT", "TOTAL TRAFFICKED", f"Trafficking in {state}"), use_container_width=True)

# ------------------ FOOTER ------------------
st.markdown("---")
st.caption("🧠 Built with ❤️ using Streamlit, Plotly & Machine Learning | India Crime Project © Ayushman Kar")
>>>>>>> b62ab8d (Initial commit for Streamlit deployment)
