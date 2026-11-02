"""
NANTENAINA MIALY ANDRIANARIVONY
Streamlit Interactive ML Web Dashboard Integration
"""
#IMPORTS
import joblib
import pandas as pd
import streamlit as st
import seaborn as sns
from matplotlib import pyplot as plt

# Cache the CSV file for streamlit to load
@st.cache_data
def load_data():
    return pd.read_csv('Week6/assets/cleaned_ev_ml_dataset.csv')

# Load the ML model using joblib
@st.cache_resource
def load_model():
    return joblib.load('Week6/assets/ev_overload.joblib')

@st.cache_resource
def load_scaler():
    return joblib.load('Week6/assets/ev_overload_scaler.joblib')

# Load data into df
df = load_data()
# Access ML model
model = load_model()
# Access scaler for model
scaler = load_scaler()


def visual_page_run():
    """
    Page to display the Matplotlib visualisation charts
    4 charts to be shown: Histogram, boxplot, Scatterplot and heatmap
    Every 2 charts get put into 1 column with their respective titles above
    :return: st Page
    """
    st.title("Visual Analytics")
    st.markdown("Matplotlib charts renders")

    # Configure a dark background
    plt.style.use("dark_background")
    sns.set_palette("bright")
    sns.set_style = "ticks"
    sns.set_context("talk")
    sns.despine()

    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)

    with col1:
        st.subheader("Distribution of Energy_kWh")
        # Distribution histogram with KDE curve of 'Energy_kWh'
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        sns.histplot(data=df,
                     x="Energy_kWh",
                     kde=True,
                     ax=ax1)
        ax1.set_xlabel("Energy Delivered (kWh)")
        st.pyplot(fig1)

    with col2:
         # Boxplot comparing 'Duration_Mins' across 'Charger_Type' (AC vs DC).
        st.subheader("Boxplot - duration across charger types")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=df,
                    x="Duration_Mins",
                    y="Charger_Type",
                    hue="Charger_Type",
                    ax=ax2)
        ax2.set_xlabel("Duration in Minutes")
        ax2.set_ylabel("Charger Type")
        st.pyplot(fig2)

    with col3:
        # Scatterplot of 'Duration_Mins' vs 'Energy_kWh' hue-coded by 'User_Type'
        st.subheader("Scatterplot - duration across energy")
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=df,
                        x="Duration_Mins",
                        y="Energy_kWh",
                        hue="User_Type",
                        ax=ax3)
        ax3.set_xlabel("Duration in Minutes")
        ax3.set_ylabel("Energy Delivered (kWh)")
        st.pyplot(fig3)

    with col4:
        # Seaborn heatmap showing correlation matrix between numeric telemetry features.
        st.subheader("Heatmap - Correlation matrix")
        fig4, ax4 = plt.subplots(figsize=(15, 8))
        corr_matrix = df.corr(numeric_only=True)
        sns.heatmap(data=corr_matrix,
                    annot=True,
                    ax=ax4)
        sns.color_palette("rocket", as_cmap=True)
        ax3.set_xlabel("Duration in Minutes")
        ax3.set_ylabel("Energy Delivered (kWh)")
        st.pyplot(fig4)


def ml_page_run():
    """
    Renders 2 columns
    Column 1 to input parameters for the prediction
    All data is then converted into a pandas Dataframe (in the order they appear in X features)
    Then scaled using loaded scaler() and passed to .predict()

    Column 2 is the prediction result

    :return: st Page
    """
    st.title("Live Machine Learning Inference")
    st.markdown("Adjust the parameters below to predict the real-time overload risk for a session.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("⚙️ Input Parameters Below:")
        st.write("Please make sure that all parameters are entered correctly.")
        start_hour = st.slider("Start Hour (0 - 24)", min_value=0, max_value=24, step=1)

        duration_mins = st.number_input("Duration (Minutes) [15.0 - 300.0]",
                                        min_value=15.0, max_value=300.0, value=15.0, step=0.01)

        energy_kwh = st.number_input("Energy Delivered (kWh) [5.0 - 120.0]",
                                     min_value=5.0, max_value=120.0, value=5.0, step=0.01)

        overstay_mins = st.number_input("Overstay (in minutes)[0 - 300]"
                                        , min_value=0, max_value=300, value=0, step=1)

        battery_temp = st.slider("Battery Temperature (°C) [24.0 - 70.0]",
                                 min_value=24.0, max_value=70.0, step=0.1)

        power_kw = st.slider("Power in kW [0 - 70.0]",
                                 min_value=0.0, max_value=70.0, step=0.01)

        overstay_ratio = st.slider("Overstay Ratio [0 - 1.5]",
                                 min_value=0.0, max_value=1.5, step=0.01)

        charger_type = st.selectbox("Charger Type", ["AC_Fast", "DC_Ultra"])

        grid_status_label = st.selectbox("Grid Status Label",
                                         ["Normal", "Overheated", "Grid Surge"])

        # Convert data into matching one hot encode type
        charger_ac = 0
        charger_dc = 0
        if charger_type == "AC_Fast":
            charger_ac = 1
        if charger_type == "DC_Ultra":
            charger_dc = 1

        grid_normal = 0
        grid_overheated = 0
        grid_surge = 0
        if grid_status_label == "Normal":
            grid_normal = 1
        if grid_status_label == "Overheated":
            grid_overheated = 1
        if grid_status_label == "Grid Surge":
            grid_surge = 1

        # Button to call predict()
        predict_btn = st.button("🚀 Predict Grid Status", use_container_width=True)


    with col2:
        st.subheader("📊 Prediction Result")

        # When clicked
        if predict_btn:
            #['Start_Hour', 'Duration_Mins', 'Energy_kWh', 'Overstay_Mins',
             #        'Battery_Temp_C', 'Power_kW', 'Overstay_Ratio', 'Charger_Type_AC',
             #        'Charger_Type_DC', 'Grid_Status_Label_Grid_Surge',
             #        'Grid_Status_Label_Normal', 'Grid_Status_Label_Overheated']
            # Convert all parameters into a pandas DataFrame that match the required features for the prediction
            input_data = pd.DataFrame([{
                'Start_Hour': start_hour,
                'Duration_Mins': duration_mins,
                'Energy_kWh': energy_kwh,
                'Overstay_Mins': overstay_mins,
                'Battery_Temp_C': battery_temp,
                'Power_kW': power_kw,
                'Overstay_Ratio': overstay_ratio,
                'Charger_Type_AC': charger_ac,
                'Charger_Type_DC': charger_dc,
                'Grid_Status_Label_Grid_Surge': grid_surge,
                'Grid_Status_Label_Normal': grid_normal,
                'Grid_Status_Label_Overheated': grid_overheated

            }])

            # Make prediction using the loaded model
            scaled_input_data = scaler.transform(input_data)
            prediction = model.predict(scaled_input_data)

            # # Calculate power and carbon savings based on formula in the data_pipeline
            # st.write(f"Based on the inputs, the calculated **Power** is: `{energy_kwh:.2f} kW`")
            # st.write(f"**Carbon Savings** is: `{carbon_savings:.2f} kg C02`")

            st.divider()

            # From the y_pred
            # if model predicts that the inputs are `Normal`, display balloons on screen
            if prediction == 0:
                st.success(
                    f"✅ **Predicted Risk: {prediction} NO**\n\n"
                    f"No Overload Risk. No action required.")
                st.balloons()  # Balloon animation
            # if model predicts that the inputs are `Overheated`, create st.warning
            elif prediction == 1:
                st.error(
                    f"🚨 **Predicted Risk: {prediction} YES**\n\n"
                    f"Warning: POSSIBLE OVERLOAD RISK. Please monitor the session.")
            else:
                st.info(f"Predicted Status: {prediction}")
        else:
            st.info("👈 Please adjust the parameters on the left and click 'Predict Grid Status'.")

# MAIN PAGE STARTS HERE
st.set_page_config(page_title="Dashboard", layout="wide")

visual_page = st.Page(visual_page_run,  title="Visual Analytics", icon="ℹ️")
ml_page = st.Page(ml_page_run, title="Predictions", icon="📊")

pg = st.navigation([visual_page, ml_page])

st.set_page_config(page_title="Data manager", page_icon=":material/edit:")

pg.run()
