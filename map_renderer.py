import folium
from streamlit_folium import st_folium

def render_disease_map(logs):
    m = folium.Map(location=[10.5, 7.5], zoom_start=6)  # Nigeria center

    for log in logs:
        popup_text = f"{log['disease']}<br>{log['timestamp']}"
        folium.Marker(
            location=[log["latitude"], log["longitude"]],
            popup=popup_text,
            icon=folium.Icon(color="red" if log["disease"] != "Healthy" else "green")
        ).add_to(m)

    st_folium(m, width=700, height=500)
