import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Configuración de la página
st.set_page_config(page_title="Sismos Santander", page_icon="📡", layout="wide")

st.title("📡 Sistema de Estimación de Actividad Sísmica - Santander")
st.markdown("Esta aplicación experimental estima el nivel de actividad sísmica basándose en datos 100% reales. **Nota:** No predice terremotos con exactitud.")
st.divider()

# ==========================================
# 1. DICCIONARIO DE UBICACIONES 
# ==========================================
UBICACIONES = {
    "Los Santos (Nido Sísmico)": {"lat": 6.766, "lon": -73.104},
    "Bucaramanga": {"lat": 7.119, "lon": -73.122},
    "Floridablanca": {"lat": 7.062, "lon": -73.086},
    "San Gil": {"lat": 6.559, "lon": -73.133},
    "Barrancabermeja": {"lat": 7.065, "lon": -73.854},
    "Socorro": {"lat": 6.467, "lon": -73.260},
    "Zapatoca": {"lat": 6.814, "lon": -73.268},
    "Piedecuesta": {"lat": 6.987, "lon": -73.050}
}

# ==========================================
# 2. PANEL PRINCIPAL: SELECCIÓN
# ==========================================
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📍 Selecciona la ubicación")
    lugar_seleccionado = st.selectbox("Elige un municipio o zona:", list(UBICACIONES.keys()))
    
    lat_seleccionada = UBICACIONES[lugar_seleccionado]["lat"]
    lon_seleccionada = UBICACIONES[lugar_seleccionado]["lon"]
    st.info(f"Coordenadas base: Latitud {lat_seleccionada} | Longitud {lon_seleccionada}")

# ==========================================
# 3. CARGA DE DATOS REALES Y FILTRADO
# ==========================================
df_local = None
if os.path.exists('datos_sismos.csv'):
    df = pd.read_csv('datos_sismos.csv')
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    
    # Filtrar solo los sismos cercanos al municipio seleccionado (radio de ~50km)
    margen = 0.5 
    df_local = df[(df['latitude'] >= lat_seleccionada - margen) & 
                  (df['latitude'] <= lat_seleccionada + margen) & 
                  (df['longitude'] >= lon_seleccionada - margen) & 
                  (df['longitude'] <= lon_seleccionada + margen)].copy()

# ==========================================
# 4. PREDICCIÓN CON VARIABLES CALCULADAS EN TIEMPO REAL
# ==========================================
with col2:
    st.subheader("🔮 Estimación del Modelo")
    
    try:
        datos_guardados = joblib.load('modelo_sismos.joblib')
        modelo = datos_guardados.get('modelo', datos_guardados.get('model')) if isinstance(datos_guardados, dict) else datos_guardados
        
        if df_local is not None:
            st.success(f"✅ ¡Conectado! Se encontraron {len(df_local)} sismos reales cerca de {lugar_seleccionado}.")
            
            if st.button("Estimar Actividad para la Próxima Semana", type="primary", use_container_width=True):
                nombres_esperados = getattr(modelo, 'feature_names_in_', None)
                
                if nombres_esperados is not None:
                    # Tomamos los últimos 30 días de la zona seleccionada para calcular la actividad reciente
                    fecha_max = df_local['time'].max()
                    df_reciente = df_local[df_local['time'] >= (fecha_max - pd.Timedelta(days=30))]
                    
                    # Diccionario automático para calcular lo que el modelo pida
                    estadisticas = {
                        'n_eventos': len(df_reciente),
                        'mag_media': df_reciente['mag'].mean() if not df_reciente.empty else 0,
                        'mag_max': df_reciente['mag'].max() if not df_reciente.empty else 0,
                        'mag_std': df_reciente['mag'].std() if len(df_reciente) > 1 else 0,
                        'prof_media': df_reciente['depth'].mean() if not df_reciente.empty else 0,
                        'prof_min': df_reciente['depth'].min() if not df_reciente.empty else 0,
                        'prof_max': df_reciente['depth'].max() if not df_reciente.empty else 0,
                        'energia': df_reciente['energia'].sum() if ('energia' in df_reciente.columns and not df_reciente.empty) else 0,
                        'energia_total': df_reciente['energia'].sum() if ('energia' in df_reciente.columns and not df_reciente.empty) else 0,
                        'lat_celda': lat_seleccionada,
                        'lon_celda': lon_seleccionada
                    }
                    
                    # Construimos el dato de entrada exacto que pide tu modelo
                    datos_entrada = pd.DataFrame([{col: estadisticas.get(col, 0) for col in nombres_esperados}])
                    datos_entrada = datos_entrada.fillna(0)
                    
                    # Predicción con el modelo
                    prediccion = modelo.predict(datos_entrada)
                    
                    st.write("### Nivel Estimado:")
                    if prediccion[0] == 'Baja' or prediccion[0] == 0:
                        st.success("🟢 Actividad BAJA estimada (0–15 sismos esperados).")
                    elif prediccion[0] == 'Media' or prediccion[0] == 1:
                        st.warning("🟡 Actividad MEDIA estimada (16–50 sismos esperados).")
                    else:
                        st.error("🔴 Actividad ALTA estimada (Más de 50 sismos esperados).")
                else:
                    st.error("El modelo no tiene guardados los nombres de las variables.")
        else:
            st.error("⚠️ No se encuentra el archivo 'datos_sismos.csv'.")
    except Exception as e:
        st.error(f"Error al calcular la predicción: {e}")

st.divider()

# ==========================================
# 5. GRÁFICAS Y MAPAS CON DATOS REALES
# ==========================================
st.subheader("📊 Análisis Visual e Histórico (Datos Reales)")

col_mapa, col_graficas = st.columns([1, 1])

with col_mapa:
    st.markdown(f"**Ubicación de los sismos cerca a {lugar_seleccionado}**")
    if df_local is not None and not df_local.empty:
        df_mapa = df_local[['latitude', 'longitude']].rename(columns={'latitude': 'lat', 'longitude': 'lon'})
        st.map(df_mapa, zoom=8)

with col_graficas:
    if df_local is not None and not df_local.empty:
        st.markdown("**Evolución de Magnitudes Máximas (Mensual)**")
        df_grafica = df_local.set_index('time')['mag'].sort_index()
        df_grafica_mensual = df_grafica.resample('ME').max().dropna()
        st.line_chart(df_grafica_mensual)
        
        st.markdown("**Distribución de sismos por Profundidad**")
        etiquetas = ['Superficial (0-30km)', 'Media (30-70km)', 'Profunda (>70km)']
        df_local['Categoria_Prof'] = pd.cut(df_local['depth'], bins=[0, 30, 70, 3000], labels=etiquetas)
        conteo_prof = df_local['Categoria_Prof'].value_counts().sort_index()
        st.bar_chart(conteo_prof)

st.divider()

# ==========================================
# 6. TABLA DE DATOS CONSULTADOS REALES
# ==========================================
st.subheader("📋 Catálogo Sísmico Consultado")
if df_local is not None and not df_local.empty:
    st.markdown(f"Registros históricos asociados a la zona de **{lugar_seleccionado}**.")
    df_mostrar = df_local.sort_values(by='time', ascending=False)
    columnas_mostrar = [col for col in ['time', 'latitude', 'longitude', 'depth', 'mag', 'energia'] if col in df_mostrar.columns]
    st.dataframe(df_mostrar[columnas_mostrar], use_container_width=True, hide_index=True)