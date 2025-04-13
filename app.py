import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import io
import base64
import sqlite3

# Configuración inicial de la página
st.set_page_config(
    page_title="Análisis Interactivo de Datos",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Aplicar un tema personalizado
def local_css():
    st.markdown("""
        <style>
        .stApp {
            background-color: #f5f5f5;
        }
        .main .block-container {
            padding-top: 2rem;
        }
        h1, h2, h3 {
            color: #2c3e50;
        }
        .sidebar .sidebar-content {
            background-color: #f0f2f6;
        }
        /* Estilo para el fondo oscuro/gris en la sección principal */
        .dark-section {
            background-color: #2c3e50; /* Color azul oscuro */
            color: white; /* Texto blanco para contraste */
            padding: 1rem;
            border-radius: 8px; /* Bordes redondeados */
            margin-bottom: 1rem;
        }
        /* Estilo para el contenedor de información del dataset */
        .info-section {
            background-color: #34495e; /* Color gris azulado */
            color: white;
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
        }
        /* Estilo para secciones con fondo blanco y texto negro */
        .white-section {
            background-color: white;
            color: black; /* Texto negro para contraste */
        }
        /* Estilo para la tabla de datos */
        .data-table {
            background-color: white; /* Fondo blanco */
            color: black; /* Texto negro */
            border: 1px solid #ddd;
            border-collapse: collapse;
            width: 100%;
        }
        .data-table th, .data-table td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        .data-table th {
            background-color: #f0f2f6; /* Encabezados de columna ligeramente más claros */
            font-weight: bold;
        }
        </style>
    """, unsafe_allow_html=True)

local_css()

# Función para inicializar la base de datos
def init_db():
    conn = sqlite3.connect('data_storage.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS datasets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_name TEXT,
            data BLOB
        )
    ''')
    conn.commit()
    conn.close()

# Función para guardar un DataFrame en la base de datos
def save_to_db(dataset_name, df):
    conn = sqlite3.connect('data_storage.db')
    cursor = conn.cursor()
    # Convertir el DataFrame a bytes usando pickle
    df_bytes = io.BytesIO()
    df.to_pickle(df_bytes)
    df_bytes.seek(0)
    cursor.execute('''
        INSERT INTO datasets (dataset_name, data) VALUES (?, ?)
    ''', (dataset_name, df_bytes.getvalue()))
    conn.commit()
    conn.close()

# Función para cargar un DataFrame desde la base de datos
def load_from_db(dataset_name):
    conn = sqlite3.connect('data_storage.db')
    cursor = conn.cursor()
    cursor.execute('SELECT data FROM datasets WHERE dataset_name = ?', (dataset_name,))
    result = cursor.fetchone()
    conn.close()
    if result:
        df_bytes = io.BytesIO(result[0])
        return pd.read_pickle(df_bytes)
    return None

# Cache para funciones que procesan datos
@st.cache_data
def load_data(file):
    """Carga un archivo CSV o Excel."""
    try:
        if file.name.endswith('.csv'):
            return pd.read_csv(file)
        elif file.name.endswith(('.xlsx', '.xls')):
            return pd.read_excel(file)
        else:
            st.error("Formato de archivo no soportado. Solo se admiten archivos CSV o Excel.")
            return None
    except Exception as e:
        st.error(f"Error al cargar el archivo: {e}")
        return None

@st.cache_data
def get_csv_download_link(df, filename="datos_exportados.csv"):
    """Genera un enlace de descarga para un DataFrame."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Descargar CSV</a>'
    return href

def generate_profile_report(df):
    """Genera un reporte de perfil usando pandas-profiling."""
    from pandas_profiling import ProfileReport
    profile = ProfileReport(df, title="Reporte de Análisis Exploratorio", explorative=True)
    return profile.to_html()

def main():
    with st.sidebar:
        st.image("https://via.placeholder.com/150x50?text=DataViz", width=150)
        st.title("DataViz Pro")
        st.markdown("---")

        # Sección principal
        st.title("📊 Aplicación Avanzada de Visualización de Datos")
        st.markdown('<div class="dark-section">', unsafe_allow_html=True)
        st.write("""
        Bienvenido/a a DataViz Pro, una herramienta completa para el análisis y visualización 
        interactiva de datos. Carga tu dataset y descubre insights valiosos con nuestras 
        múltiples funcionalidades.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    # ------------------------------------------
    # 1. Carga de Datasets
    # ------------------------------------------
    with st.sidebar.expander("📁 CARGAR DATOS", expanded=True):
        file = st.file_uploader("Sube un archivo CSV o Excel", type=["csv", "xlsx", "xls"])
        demo_data = st.checkbox("¿Usar datos de demostración?")
        if demo_data:
            option = st.selectbox(
                "Selecciona un dataset de ejemplo:",
                ["Iris", "Titanic", "Diamantes"]
            )
            if option == "Iris":
                df = sns.load_dataset("iris")
            elif option == "Titanic":
                df = sns.load_dataset("titanic")
            else:
                df = sns.load_dataset("diamonds").sample(1000)  # Muestra para rendimiento
            st.success(f"Dataset de demostración '{option}' cargado!")
        elif file is not None:
            with st.spinner("Cargando datos..."):
                df = load_data(file)
                if df is not None:
                    st.success(f"¡Archivo '{file.name}' cargado exitosamente!")
        else:
            st.info("👆 Por favor, carga un archivo o usa un dataset de demostración.")
            df = None

    # Continuar solo si hay datos cargados
    if df is not None:
        # ------------------------------------------
        # 2. Análisis Exploratorio
        # ------------------------------------------
        st.sidebar.markdown("---")
        with st.sidebar.expander("🔍 ANÁLISIS Y FILTROS"):
            # Opciones de limpieza de datos
            if st.checkbox("Eliminar filas con valores nulos"):
                df = df.dropna()
                st.success(f"Se eliminaron las filas con valores nulos. Nuevo tamaño: {df.shape}")

            # Filtros dinámicos
            st.subheader("Filtros Dinámicos")
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

            # Filtrado por columnas categóricas
            if categorical_cols:
                selected_cat_cols = st.multiselect("Filtrar por categorías:", options=categorical_cols)
                for col in selected_cat_cols:
                    unique_values = df[col].unique()
                    selected_values = st.multiselect(
                        f"Valores para {col}:",
                        options=unique_values,
                        default=list(unique_values)[:5] if len(unique_values) > 5 else list(unique_values)
                    )
                    if selected_values:
                        df = df[df[col].isin(selected_values)]

            # Filtrado por columnas numéricas
            if numeric_cols:
                selected_num_cols = st.multiselect("Filtrar por rangos numéricos:", options=numeric_cols)
                for col in selected_num_cols:
                    min_val, max_val = float(df[col].min()), float(df[col].max())
                    range_vals = st.slider(
                        f"Rango para {col}:",
                        min_value=min_val,
                        max_value=max_val,
                        value=(min_val, max_val)
                    )
                    df = df[(df[col] >= range_vals[0]) & (df[col] <= range_vals[1])]

        # Interfaz principal con pestañas
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📋 Vista Previa", 
            "📊 Visualizaciones Básicas", 
            "🔍 Visualizaciones Avanzadas",
            "📈 Series Temporales",
            "⚙️ Exportar y Compartir"
        ])

        # ------------------------------------------
        # TAB 1: Vista Previa y Estadísticas Básicas
        # ------------------------------------------
        with tab1:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader("Vista Previa del Dataset")
                st.markdown('<div class="white-section">', unsafe_allow_html=True)
                st.dataframe(df.head(10), use_container_width=True)
                show_all = st.checkbox("Ver todos los datos")
                if show_all:
                    st.dataframe(df, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                st.subheader("Información del Dataset")
                st.markdown('<div class="info-section">', unsafe_allow_html=True)
                st.write(f"**Filas**: {df.shape[0]} | **Columnas**: {df.shape[1]}")
                # Información sobre tipos de datos
                st.write("##### Tipos de Datos")
                dtypes_df = pd.DataFrame(df.dtypes, columns=["Tipo de Dato"])
                dtypes_df = dtypes_df.reset_index().rename(columns={"index": "Columna"})
                st.dataframe(dtypes_df.style.set_table_attributes('class="data-table white-section"'), use_container_width=True)
                # Información sobre valores nulos
                st.write("##### Valores Nulos")
                null_counts = df.isnull().sum()
                null_df = pd.DataFrame({
                    "Columna": null_counts.index,
                    "Valores Nulos": null_counts.values,
                    "% Nulos": (null_counts.values / len(df) * 100).round(2)
                })
                st.dataframe(null_df.style.set_table_attributes('class="data-table white-section"'), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

        # ------------------------------------------
        # TAB 2: Visualizaciones Básicas
        # ------------------------------------------
        with tab2:
            st.subheader("Visualizaciones Básicas")
            viz_type = st.radio(
                "Selecciona el tipo de visualización:",
                ["Gráfico de Barras", "Gráfico de Dispersión", "Gráfico de Líneas", "Gráfico de Caja", "Mapa de Calor"]
            )

            if viz_type == "Gráfico de Barras":
                create_bar_chart(df, categorical_cols, numeric_cols)
            elif viz_type == "Gráfico de Dispersión":
                create_scatter_plot(df, numeric_cols, categorical_cols)
            elif viz_type == "Gráfico de Líneas":
                create_line_chart(df, numeric_cols)
            elif viz_type == "Gráfico de Caja":
                create_box_plot(df, numeric_cols, categorical_cols)
            elif viz_type == "Mapa de Calor":
                create_heatmap(df, numeric_cols)

        # ------------------------------------------
        # TAB 3: Visualizaciones Avanzadas
        # ------------------------------------------
        with tab3:
            st.subheader("Visualizaciones Avanzadas")
            adv_viz_type = st.selectbox(
                "Selecciona el tipo de visualización avanzada:",
                ["Gráficos de Violín", "Gráficos de Densidad", "Gráficos de Burbujas", "PCA (Análisis de Componentes Principales)", "Gráfico Sunburst", "Gráfico de Paralelas"]
            )

            if adv_viz_type == "Gráficos de Violín":
                create_violin_plot(df, numeric_cols, categorical_cols)
            elif adv_viz_type == "Gráficos de Densidad":
                create_density_plot(df, numeric_cols, categorical_cols)
            elif adv_viz_type == "Gráficos de Burbujas":
                create_bubble_chart(df, numeric_cols, categorical_cols)
            elif adv_viz_type == "PCA (Análisis de Componentes Principales)":
                perform_pca(df, numeric_cols, categorical_cols)

        # ------------------------------------------
        # TAB 4: Series Temporales
        # ------------------------------------------
        with tab4:
            st.subheader("Series Temporales")
            st.warning("Esta funcionalidad aún está en desarrollo.")

        # ------------------------------------------
        # TAB 5: Exportar y Compartir
        # ------------------------------------------
        with tab5:
            st.subheader("Exportar Datos")
            st.markdown(get_csv_download_link(df), unsafe_allow_html=True)

    if df is not None:
    # Guardar el DataFrame en la base de datos
    dataset_name = file.name if file else "demo_dataset"
    save_to_db(dataset_name, df)
    st.success(f"Dataset '{dataset_name}' guardado en la base de datos.")

def create_bar_chart(df, categorical_cols, numeric_cols):
    """Crea un gráfico de barras interactivo."""
    if categorical_cols and numeric_cols:
        col1, col2 = st.columns(2)
        with col1:
            cat_col_bar = st.selectbox("Columna Categórica (Eje X)", options=categorical_cols)
            num_col_bar = st.selectbox("Columna Numérica (Eje Y)", options=numeric_cols)
            agg_func = st.selectbox("Función de Agregación", options=["Media", "Suma", "Mediana", "Mínimo", "Máximo", "Contar"], index=0)
            agg_dict = {"Media": "mean", "Suma": "sum", "Mediana": "median", "Mínimo": "min", "Máximo": "max", "Contar": "count"}
            agg_data = df.groupby(cat_col_bar)[num_col_bar].agg(agg_dict[agg_func]).reset_index()
            agg_data = agg_data.sort_values(num_col_bar, ascending=False).head(15)

        with col2:
            orientation = st.radio("Orientación:", ["Vertical", "Horizontal"])
            color_theme = st.selectbox("Esquema de Color:", options=["viridis", "plasma", "inferno", "magma", "cividis"])

        if orientation == "Vertical":
            fig = px.bar(agg_data, x=cat_col_bar, y=num_col_bar, title=f"{agg_func} de {num_col_bar} por {cat_col_bar}", color=cat_col_bar, color_continuous_scale=color_theme)
        else:
            fig = px.bar(agg_data, y=cat_col_bar, x=num_col_bar, title=f"{agg_func} de {num_col_bar} por {cat_col_bar}", color=cat_col_bar, color_continuous_scale=color_theme, orientation='h')

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Se necesitan columnas categóricas y numéricas para crear un gráfico de barras.")

def create_scatter_plot(df, numeric_cols, categorical_cols):
    """Crea un gráfico de dispersión interactivo."""
    if len(numeric_cols) >= 2:
        col1, col2 = st.columns(2)
        with col1:
            x_axis = st.selectbox("Eje X", numeric_cols, index=0)
            y_axis = st.selectbox("Eje Y", numeric_cols, index=min(1, len(numeric_cols)-1))
            color_var = None
            size_var = None
            if categorical_cols:
                color_var = st.selectbox("Variable para Color (opcional)", options=["Ninguna"] + categorical_cols)
                if color_var == "Ninguna":
                    color_var = None

        with col2:
            if numeric_cols:
                size_var = st.selectbox("Variable para Tamaño (opcional)", options=["Ninguna"] + numeric_cols)
                if size_var == "Ninguna":
                    size_var = None
            add_trendline = st.checkbox("Añadir línea de tendencia")

        scatter_fig = px.scatter(
            df, x=x_axis, y=y_axis, color=color_var, size=size_var, trendline="ols" if add_trendline else None,
            title=f"Gráfico de Dispersión: {x_axis} vs {y_axis}"
        )
        st.plotly_chart(scatter_fig, use_container_width=True)
    else:
        st.warning("Se necesitan al menos dos columnas numéricas para crear un gráfico de dispersión.")

def create_line_chart(df, numeric_cols):
    """Crea un gráfico de líneas interactivo."""
    if numeric_cols:
        col1, col2 = st.columns(2)
        with col1:
            x_axis_line = st.selectbox("Eje X (Línea)", options=df.columns.tolist())
            y_axes_line = st.multiselect("Ejes Y (pueden ser múltiples)", options=numeric_cols, default=[numeric_cols[0]])

        with col2:
            line_mode = st.radio("Modo de Línea:", ["Líneas", "Marcadores", "Líneas + Marcadores"])
            mode_dict = {"Líneas": "lines", "Marcadores": "markers", "Líneas + Marcadores": "lines+markers"}

        if y_axes_line:
            line_fig = go.Figure()
            for y_col in y_axes_line:
                sorted_df = df.sort_values(by=x_axis_line)
                line_fig.add_trace(go.Scatter(x=sorted_df[x_axis_line], y=sorted_df[y_col], mode=mode_dict[line_mode], name=y_col))

            line_fig.update_layout(title=f"Gráfico de Líneas: {x_axis_line} vs {', '.join(y_axes_line)}", xaxis_title=x_axis_line, yaxis_title="Valores", legend_title="Variables")
            st.plotly_chart(line_fig, use_container_width=True)
        else:
            st.warning("Selecciona al menos una columna para el eje Y.")
    else:
        st.warning("Se necesitan columnas numéricas para crear un gráfico de líneas.")

def create_box_plot(df, numeric_cols, categorical_cols):
    """Crea un gráfico de caja interactivo."""
    if numeric_cols:
        col1, col2 = st.columns(2)
        with col1:
            y_box = st.selectbox("Variable Numérica", options=numeric_cols)
            x_box = None
            if categorical_cols:
                x_box = st.selectbox("Variable Categórica (opcional)", options=["Ninguna"] + categorical_cols)
                if x_box == "Ninguna":
                    x_box = None

        with col2:
            orientation = st.radio("Orientación (Caja):", ["Vertical", "Horizontal"])
            add_points = st.checkbox("Mostrar puntos individuales", value=True)

        box_fig = px.box(
            df, y=y_box if orientation == "Vertical" else None, x=y_box if orientation == "Horizontal" else None, color=x_box,
            category_orders={x_box: sorted(df[x_box].unique()) if x_box else None}, points="all" if add_points else "outliers",
            title=f"Gráfico de Caja: {y_box}" + (f" por {x_box}" if x_box else "")
        )
        st.plotly_chart(box_fig, use_container_width=True)
    else:
        st.warning("Se necesitan columnas numéricas para crear un gráfico de caja.")

def create_heatmap(df, numeric_cols):
    """Crea un mapa de calor de correlaciones."""
    if len(numeric_cols) >= 2:
        st.subheader("Mapa de Calor de Correlaciones")
        corr_method = st.radio("Método de Correlación:", ["Pearson", "Spearman", "Kendall"]).lower()
        corr_matrix = df[numeric_cols].corr(method=corr_method)

        heatmap_fig = px.imshow(
            corr_matrix, text_auto=True, aspect="auto", color_continuous_scale=px.colors.diverging.RdBu_r,
            title=f"Matriz de Correlación ({corr_method.capitalize()})"
        )
        st.plotly_chart(heatmap_fig, use_container_width=True)

        with st.expander("Ver tabla de correlaciones"):
            st.dataframe(corr_matrix, use_container_width=True)
    else:
        st.warning("Se necesitan al menos dos columnas numéricas para crear un mapa de calor.")

def create_violin_plot(df, numeric_cols, categorical_cols):
    """Crea un gráfico de violín interactivo."""
    if numeric_cols and categorical_cols:
        col1, col2 = st.columns(2)
        with col1:
            y_violin = st.selectbox("Variable Numérica (Violín)", options=numeric_cols)
            x_violin = st.selectbox("Variable Categórica (Violín)", options=categorical_cols)

        with col2:
            box_inside = st.checkbox("Mostrar caja interna", value=True)
            violin_orientation = st.radio("Orientación (Violín):", ["Vertical", "Horizontal"])

        violin_fig = px.violin(
            df, y=y_violin if violin_orientation == "Vertical" else None, x=y_violin if violin_orientation == "Horizontal" else None,
            color=x_violin, box=box_inside, points="all", title=f"Gráfico de Violín: {y_violin} por {x_violin}"
        )
        st.plotly_chart(violin_fig, use_container_width=True)
    else:
        st.warning("Se necesitan columnas numéricas y categóricas para crear un gráfico de violín.")

def create_density_plot(df, numeric_cols, categorical_cols):
    """Crea un gráfico de densidad interactivo."""
    if numeric_cols:
        col1, col2 = st.columns(2)
        with col1:
            dens_vars = st.multiselect("Variables Numéricas para Densidad", options=numeric_cols, default=[numeric_cols[0]])

        with col2:
            color_var_dens = None
            if categorical_cols:
                color_var_dens = st.selectbox("Variable para Color (Densidad)", options=["Ninguna"] + categorical_cols)
                if color_var_dens == "Ninguna":
                    color_var_dens = None

        if dens_vars:
            if not color_var_dens:
                dens_fig = go.Figure()
                for var in dens_vars:
                    dens_fig.add_trace(go.Violin(x=df[var], name=var, side="positive", orientation="h"))
                dens_fig.update_layout(title="Gráfico de Densidad", yaxis_title="Variables", xaxis_title="Valores")
                st.plotly_chart(dens_fig, use_container_width=True)
            else:
                for var in dens_vars:
                    kde_fig = px.histogram(df, x=var, color=color_var_dens, marginal="rug", opacity=0.7, histnorm="probability density", title=f"Densidad de {var} por {color_var_dens}")
                    st.plotly_chart(kde_fig, use_container_width=True)
        else:
            st.warning("Selecciona al menos una variable numérica.")
    else:
        st.warning("Se necesitan columnas numéricas para crear gráficos de densidad.")

def create_bubble_chart(df, numeric_cols, categorical_cols):
    """Crea un gráfico de burbujas interactivo."""
    if len(numeric_cols) >= 3:
        col1, col2 = st.columns(2)
        with col1:
            x_bubble = st.selectbox("Eje X (Burbujas)", options=numeric_cols, index=0)
            y_bubble = st.selectbox("Eje Y (Burbujas)", options=numeric_cols, index=min(1, len(numeric_cols)-1))
            size_bubble = st.selectbox("Tamaño de Burbujas", options=numeric_cols, index=min(2, len(numeric_cols)-1))

        with col2:
            color_bubble = None
            if categorical_cols:
                color_bubble = st.selectbox("Color (Burbujas)", options=["Ninguna"] + categorical_cols)
                if color_bubble == "Ninguna":
                    color_bubble = None
            hover_data = st.multiselect("Datos adicionales al pasar el ratón", options=[col for col in df.columns if col not in [x_bubble, y_bubble, size_bubble, color_bubble]], default=[])

        bubble_fig = px.scatter(
            df, x=x_bubble, y=y_bubble, size=size_bubble, color=color_bubble, hover_name=df.index, hover_data=hover_data,
            title=f"Gráfico de Burbujas: {x_bubble} vs {y_bubble} (Tamaño: {size_bubble})"
        )
        st.plotly_chart(bubble_fig, use_container_width=True)
    else:
        st.warning("Se necesitan al menos tres columnas numéricas para crear un gráfico de burbujas.")

def perform_pca(df, numeric_cols, categorical_cols):
    """Realiza un análisis de componentes principales (PCA)."""
    if len(numeric_cols) >= 3:
        st.subheader("Análisis de Componentes Principales (PCA)")
        col1, col2 = st.columns(2)
        with col1:
            pca_vars = st.multiselect("Variables para PCA", options=numeric_cols, default=numeric_cols[:min(5, len(numeric_cols))])
            n_components = st.slider("Número de componentes", min_value=2, max_value=min(len(pca_vars), 5), value=2)

        with col2:
            color_pca = None
            if categorical_cols:
                color_pca = st.selectbox("Variable para Color (PCA)", options=["Ninguna"] + categorical_cols)
                if color_pca == "Ninguna":
                    color_pca = None
            scaling = st.checkbox("Escalar los datos", value=True)

        if len(pca_vars) >= 2:
            pca_data = df[pca_vars].copy()
            pca_data = pca_data.fillna(pca_data.mean())
            if scaling:
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(pca_data)
            else:
                scaled_data = pca_data.values

            pca = PCA(n_components=n_components)
            components = pca.fit_transform(scaled_data)
            pca_df = pd.DataFrame(data=components, columns=[f"PC{i+1}" for i in range(n_components)])
            if color_pca:
                pca_df[color_pca] = df[color_pca].values

            if n_components >= 2:
                pca_fig = px.scatter(pca_df, x="PC1", y="PC2", color=color_pca, title=f"PCA: PC1 vs PC2 (Varianza explicada: {pca.explained_variance_ratio_[0]*100:.1f}% y {pca.explained_variance_ratio_[1]*100:.1f}%)")
                st.plotly_chart(pca_fig, use_container_width=True)

            if n_components >= 3:
                pca_3d_fig = px.scatter_3d(pca_df, x="PC1", y="PC2", z="PC3", color=color_pca, title=f"PCA 3D: PC1 vs PC2 vs PC3")
                st.plotly_chart(pca_3d_fig, use_container_width=True)

            explained_var = pd.DataFrame({
                'Componente': [f"PC{i+1}" for i in range(n_components)],
                'Varianza Explicada (%)': pca.explained_variance_ratio_ * 100,
                'Varianza Explicada Acumulada (%)': np.cumsum(pca.explained_variance_ratio_) * 100
            })
            st.subheader("Varianza Explicada por Componente")
            st.dataframe(explained_var, use_container_width=True)

            var_fig = px.bar(explained_var, x='Componente', y='Varianza Explicada (%)', text_auto='.1f', title="Varianza Explicada por Componente")
            st.plotly_chart(var_fig, use_container_width=True)
    else:
        st.warning("Se necesitan al menos tres columnas numéricas para realizar un análisis de PCA.")

if __name__ == "__main__":
    main()
