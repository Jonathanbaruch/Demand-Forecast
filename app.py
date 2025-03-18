### Streamlit

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
import openmeteo_requests
import requests_cache
from retry_requests import retry
import statsmodels.api as sm
import streamlit as st
from utils import main


### Generate Data ###
# Define parameters
start_date = datetime(2025, 1, 1)
end_date = datetime.now() + timedelta(days=6) # amount of days we have weather forecast
restaurants = ["Nulchemist", "Jordfjern", "Café Hector"]
base_seatings = [5, 8, 7]

# Call main function - but ensure it only runs once the app starts
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = True
    main(start_date, end_date, restaurants, base_seatings)  # Runs only once



### LAYOUT ###
st.set_page_config(layout="wide")  # Expands content width

# st.markdown(
#     """
#     <style>
#         /* Reduce the top margin of the sidebar */
#         [data-testid="stSidebar"] {
#             margin-top: -10px !important;  /* Adjust this value as needed */
#             padding-top: 0px !important;
#         }
#     </style>
#     """,
#     unsafe_allow_html=True
# )


st.markdown(
    """
    <style>

    /* Adjust sidebar width */
    section[data-testid="stSidebar"] {
        min-width: 280px;
        max-width: 300px;
    }
    </style>
    """,
    unsafe_allow_html=True
)



### SIDEBAR ###

# Make sidebar top margin smaller


# Sidebar: Title
st.sidebar.subheader("Restaurant Staffing")

# Sidebar: Logo
#st.sidebar.image("logo.jpg", use_container_width=True)  # Adjusts to sidebar width

# Sidebar: Choose Page
page = st.sidebar.selectbox("View", ["Forecast"])

# Sidebar: Choose Restaurant
restaurant = st.sidebar.selectbox("Restaurant", ["Nulchemist", "Jordfjern", "Café Hector"])

# Load Data
df = st.session_state.restaurant_data[restaurant]

# Ensure date is in datetime format
df["Date"] = pd.to_datetime(df["Date"])

# Create empty frame for optimal seatings
df["Optimal_Seatings"] = 0  # There is no optimal seatings, only actual

# Define date limits
min_date = df["Date"].min().date()  # Earliest available date
max_date = df["Date"].max().date() # Latest available date (set your own limit if needed)

# Sidebar: Date Selection (limited range)
selected_date = st.sidebar.date_input(
    "Date",
    value=min_date,  # Default selection
    min_value=min_date,  # Earliest allowed date
    max_value=max_date  # Latest allowed date
)

# Filter Data for Selected Date
filtered_df = df[df["Date"] == pd.to_datetime(selected_date)].copy()

# Sidebar: Employees vs. Seatings Button
if page == "Forecast":
    staffing_metric = st.sidebar.radio("Staffing_metric", ["Employees", "Seatings"], label_visibility="collapsed")



# Define column mapping based on selected metric
metric_map = {
    "Employees": {
        "Actual": "Actual_Employees",
        "Forecasted": "Forecasted_Employees",
        "Optimal": "Optimal_Employees",
        "y_axis_title": "Employees"
    },
    "Seatings": {
        "Actual": "Actual_Seatings",
        "Forecasted": "Forecast_Seatings",
        "Optimal": "Optimal_Seatings",
        "y_axis_title": "Seatings"
    }
}

# Get selected metric column names
selected_columns = metric_map[staffing_metric]






### MAIN ###


# Create two columns: main content (80%) and a small right column (20%)
col_main, col_spacer, col_side = st.columns([4, 0.5, 0.8]) 


with col_main:
    ### Plot Forecast vs. Actual vs. Optimal ###
    fig_e = go.Figure()

    # Add Actual Metric (Bars)
    fig_e.add_trace(go.Bar(
        x=filtered_df["Hour"],
        y=filtered_df[selected_columns["Actual"]],
        name=f"Actual {staffing_metric}",
        marker=dict(color="#0d0887"),
        opacity=0.8
    ))

    # Add Forecasted Metric (Bars)
    fig_e.add_trace(go.Bar(
        x=filtered_df["Hour"],
        y=filtered_df[selected_columns["Forecasted"]],
        name=f"Forecasted {staffing_metric}",
        marker=dict(color="#bd3786"),
        opacity=0.8
    ))

    # Add Optimal Metric (Bars)
    fig_e.add_trace(go.Bar(
        x=filtered_df["Hour"],
        y=filtered_df[selected_columns["Optimal"]],
        name=f"Optimal {staffing_metric}",
        marker=dict(color="#fdca26"),
        opacity=0.8
    ))

    # Update layout dynamically based on selected metric
    fig_e.update_layout(
        xaxis=dict(title="Hour", tickmode="linear"),
        yaxis=dict(title=selected_columns["y_axis_title"]),  # Dynamic Y-axis label
        barmode="group",
        bargap=0.3,
        legend=dict(title="Legend"),
        width=2000,
        height=400,
        margin=dict(t=15)
    )

    # Header:
    st.markdown(
        """
        <h2 style="text-align: center; font-size:30px; font-weight:bold; margin-bottom: 0px;">
            Shift Planner - Evening
        </h2>
        """,
        unsafe_allow_html=True
    )


    # Streamlit: Plotly Chart
    st.plotly_chart(fig_e)



    ### Error Metrics ###

    # Convert columns to numeric
    filtered_df[selected_columns["Actual"]] = pd.to_numeric(filtered_df[selected_columns["Actual"]], errors="coerce")
    filtered_df[selected_columns["Forecasted"]] = pd.to_numeric(filtered_df[selected_columns["Forecasted"]], errors="coerce")
    filtered_df[selected_columns["Optimal"]] = pd.to_numeric(filtered_df[selected_columns["Optimal"]], errors="coerce")

    if staffing_metric == "Employees":
        # Calculate Metrics
        mae_act = np.mean(np.abs(filtered_df[selected_columns["Actual"]] - filtered_df[selected_columns["Optimal"]]))
        mape_act = np.mean(np.abs((filtered_df[selected_columns["Actual"]] - filtered_df[selected_columns["Optimal"]]) / filtered_df[selected_columns["Actual"]])) * 100

        mae_fore = np.mean(np.abs(filtered_df[selected_columns["Forecasted"]] - filtered_df[selected_columns["Optimal"]]))
        mape_fore = np.mean(np.abs((filtered_df[selected_columns["Forecasted"]] - filtered_df[selected_columns["Optimal"]]) / filtered_df[selected_columns["Forecasted"]])) * 100


        # Replace NaN with 0
        mape_act = 0 if np.isnan(mape_act) else mape_act
        mape_fore = 0 if np.isnan(mape_fore) else mape_fore

        # Display Metrics
        st.write(f"**Actual Staffing: Average Error (MAPE):** {mape_act:.2f}%")
        st.write(f"**Forecasted Staffing: Average Error (MAPE) Forecasted:** {mape_fore:.2f}%")

        # Potential Savings by implementing forecast
        potential_savings = (mae_act - mae_fore) * 7  # Multiply by 7 hours open
        potential_savings = 0 if np.isnan(potential_savings) else potential_savings

        st.write(f"**Potential Hours Saved by Implementing Forecast:** {potential_savings:.2f}")
    
    else:
        # Calculate Metrics
        mae_fore = np.mean(np.abs(filtered_df[selected_columns["Forecasted"]] - filtered_df[selected_columns["Actual"]]))

        # Display Metrics
        st.write(f"**Forecasted Seatings: Average Error:** {mae_fore:.2f}")
        st.write(f"Explanation: The average difference pr. hour between the forecasted and actual seatings.")










    ### Shift Matrix ###



    # Select relevant columns

    matrix_df = df[(df["Date"].dt.date >= selected_date) & (df["Date"].dt.date <= selected_date + pd.Timedelta(days=6))].copy()
    matrix_df = matrix_df[["Date", "Hour", selected_columns["Actual"], selected_columns["Forecasted"], selected_columns["Optimal"]]]

    # Remove time from Date
    matrix_df["Date"] = matrix_df["Date"].dt.date

    # Center Date Column
    matrix_df["Date"] = matrix_df["Date"].astype(str).str.center(45)

    # Pivot the table: Each Date becomes a column, with subcolumns for employees
    matrix_pivot = matrix_df.pivot(index="Hour", columns="Date", values=[selected_columns["Actual"], selected_columns["Forecasted"], selected_columns["Optimal"]])

    # Sort columns to ensure correct order
    matrix_pivot = matrix_pivot.sort_index(axis=1, level=1)

    # Ensure Date is in top column and Employee Type in bottom column
    matrix_pivot.columns = pd.MultiIndex.from_tuples([(date, metric) for metric, date in matrix_pivot.columns])

    # Abbreviate column names for readability
    matrix_pivot.columns = matrix_pivot.columns.set_levels(["Actual", "Forecast", "Optimal"], level=1)

    ### 🚀 Apply Conditional Formatting for Multi-Index ###
    def highlight_cells(df):
        """Apply background color based on staffing levels compared to 'Optimal'."""

        # Skip if Seatings metric
        if staffing_metric == "Seatings":
            return df
        
        styles = pd.DataFrame("", index=df.index, columns=df.columns)

        for (date, metric) in df.columns:
            if metric in ["Actual", "Forecast"]:  # Only format these columns
                optimal_col = (date, "Optimal")  # Find corresponding optimal column
                if optimal_col in df.columns:  # Ensure optimal column exists
                    overstaffed = df[(date, metric)] > df[optimal_col]
                    understaffed = df[(date, metric)] < df[optimal_col]

                    # Apply colors
                    styles.loc[overstaffed, (date, metric)] = "background-color: #FFB3B3"  # Light red
                    styles.loc[understaffed, (date, metric)] = "background-color: #FFFACD"  # Yellow

        return styles

    # Apply styling function if Employees metric
    if staffing_metric == "Employees":
        styled_df = matrix_pivot.style.apply(highlight_cells, axis=None)
    else:
        styled_df = matrix_pivot.style


    # Display Header & Table
    st.markdown(
        """
        <h2 style="text-align: center; font-size:30px; font-weight:bold;">
            Shift Matrix
        </h2>
        """,
        unsafe_allow_html=True
    )

    st.dataframe(styled_df.format("{:.0f}"))







### Explainer Text ###
st.markdown(
    """
    - **Light Red**: Indicates overstaffing compared to the optimal level.
    - **Yellow**: Indicates understaffing compared to the optimal level.  \n
    """
)
st.markdown(
    """
    **How is the Optimal Staffing Level Calculated?**  
    The optimal staffing level is calculated based on the actual "busyness" for each hour.
    By deciding how many seatings staff can handle per hour, the **optimal staffing level** is determined.  
    This is then translated into the number of employees required:
    E.g. 0-3 seatings = 1 Employee, 4-8 seatings = 2 Employees, etc.
    
    """
)






### SPACER COL ###
with col_spacer:
    st.write("")  # Empty content to act as a spacer



### RIGHT COLUMN ###
# RIGHT COLUMN: Weather Widget
with col_side:
    st.markdown("### ☀️ Weather")
    st.metric(label="Temperature", value=f"{filtered_df["Temperature"].iloc[0]}°C")
    st.write(f"**Rain:** {filtered_df["Rain"].iloc[0]}mm")
    #st.write(f"**Condition:** Sunny")









