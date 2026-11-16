"""Dashboard histórico de calidad del aire en el Valle de Aburrá."""
import altair as alt
import pandas as pd
import streamlit as st

from src.agent import ask_agent
from src.data_loader import get_pm25, load_clean_data
from src.modelo_clasico import valid_reference_dates, predict_tomorrow

st.set_page_config(page_title="Calidad del aire · Valle de Aburrá", layout="wide")
st.title("Calidad del aire en el Valle de Aburrá")
st.write("Explora mediciones de PM2.5, estima la categoría del día siguiente y consulta el corpus ambiental.")

df = load_clean_data()
catalogo = df[["estacion", "municipio"]].drop_duplicates().set_index("estacion")["municipio"]
desde, hasta = df["fecha"].min().date(), df["fecha"].max().date()
st.warning("Archivo histórico de 2020. La aplicación no recibe mediciones actuales ni emite pronósticos oficiales.")
metrics = st.columns(3)
metrics[0].metric("Estaciones", len(catalogo))
metrics[1].metric("Registros", f"{len(df):,}".replace(",", "."))
metrics[2].metric("Período", f"{desde:%d/%m/%Y} a {hasta:%d/%m/%Y}")

st.caption("Datos: CSV recibido en la entrega de Persona 1. La procedencia exacta y el proceso de agregación siguen sin verificar.")
st.sidebar.header("Filtros históricos")
station = st.sidebar.selectbox("Estación", catalogo.index, format_func=lambda x: f"{x} · {catalogo[x]}")
station_rows = df.loc[df["estacion"] == station]
start = st.sidebar.date_input("Fecha inicial", station_rows["fecha"].min().date(), min_value=desde, max_value=hasta)
end = st.sidebar.date_input("Fecha final", station_rows["fecha"].max().date(), min_value=desde, max_value=hasta)
if start > end:
    st.sidebar.error("La fecha inicial debe ser anterior o igual a la final.")
latest = station_rows.iloc[-1]
st.write(f"**Último registro de {station}** · {latest['fecha']:%d/%m/%Y} · {latest['pm25']:.2f} µg/m³")
explore, compare, predict, chat = st.tabs(["Explorador", "Comparación", "Predicción", "Chat"])
with explore:
    st.header("Serie histórica de PM2.5")
    if start <= end:
        filtered = get_pm25(station, start.isoformat(), end.isoformat())
        if filtered.empty:
            st.info("No hay registros para la estación en el intervalo seleccionado.")
        else:
            # Los valores faltantes interrumpen la serie para mostrar los huecos.
            days = pd.date_range(start, end)
            series = filtered.set_index("fecha")["pm25"].reindex(days).rename_axis("fecha").reset_index()
            series["tramo"] = series["pm25"].isna().cumsum()
            marks = alt.Chart(series.dropna(subset=["pm25"])).mark_line(point=True, color="#007870")
            st.altair_chart(marks.encode(
                x=alt.X("fecha:T", title="Fecha"),
                y=alt.Y("pm25:Q", title="PM2.5 (µg/m³)"),
                detail="tramo:N",
                tooltip=[alt.Tooltip("fecha:T", title="Fecha", format="%d/%m/%Y"),
                         alt.Tooltip("pm25:Q", title="PM2.5 (µg/m³)", format=".2f")],
            ), use_container_width=True)
            st.dataframe(
                filtered[["fecha", "pm25"]].rename(columns={"fecha": "Fecha", "pm25": "PM2.5 (µg/m³)"}),
                column_config={"Fecha": st.column_config.DatetimeColumn(format="DD/MM/YYYY")},
                hide_index=True, use_container_width=True,
            )

with compare:
    st.header("PM2.5 promedio por estación")
    st.caption("El promedio usa los registros disponibles de cada estación dentro del intervalo. La cobertura puede variar.")
    means = (
        df.loc[df["fecha"].between(pd.Timestamp(start), pd.Timestamp(end))]
        .groupby(["estacion", "municipio"])["pm25"]
        .agg(promedio="mean", registros="size").reset_index()
        .sort_values("promedio", ascending=False)
    )
    if means.empty:
        st.info("El intervalo seleccionado no contiene registros para comparar.")
    else:
        means["nombre"] = means["estacion"] + " · " + means["municipio"]
        st.altair_chart(alt.Chart(means).mark_bar(color="#236e69").encode(
            y=alt.Y("nombre:N", sort="-x", title=None, axis=alt.Axis(labelLimit=400)),
            x=alt.X("promedio:Q", title="PM2.5 promedio (µg/m³)"),
            tooltip=[alt.Tooltip("nombre:N", title="Estación y municipio"),
                     alt.Tooltip("promedio:Q", title="Promedio", format=".2f"),
                     alt.Tooltip("registros:Q", title="Registros")],
        ).properties(height=max(140, len(means) * 24)), use_container_width=True)

with predict:
    st.header("Categoría experimental del día siguiente")
    st.caption("Gaussian Naive Bayes estima la probabilidad de cada categoría con tres mediciones diarias, su variación, su media móvil y el mes. El modelo ajusta distribuciones gaussianas por categoría y supone independencia condicional entre variables.")
    st.info("Predicción histórica para el día calendario siguiente. Las probabilidades del modelo no están calibradas.")
    references = valid_reference_dates(station)
    reference = None
    if references:
        reference = st.selectbox("Fecha de referencia", references[::-1], key=f"reference-{station}")
        if st.button("Ejecutar predicción", type="primary"):
            st.session_state["prediccion"] = predict_tomorrow(station, reference)
        output = st.session_state.get("prediccion")
        if output and output["estacion"] == station and output["fecha_referencia"] == reference:
            st.subheader(output["categoria_predicha"])
            st.write(f"Fecha estimada · **{output['fecha_prediccion']}**")
            st.caption(f"Ajuste con {output['n_entrenamiento']} ejemplos históricos de la estación.")
            probs = pd.DataFrame(list(output["probabilidades"].items()), columns=["Categoría", "Probabilidad"])
            st.altair_chart(alt.Chart(probs).mark_bar(color="#007870").encode(
                y=alt.Y("Categoría:N", sort=None, title=None),
                x=alt.X("Probabilidad:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
                tooltip=["Categoría:N", alt.Tooltip("Probabilidad:Q", format=".2%")],
            ), use_container_width=True)
            st.write(" · ".join(f"{c}: {p:.2%}" for c, p in output["probabilidades"].items()))
            missing = set(output["probabilidades"]) - set(output["categorias_entrenadas"])
            if missing:
                st.caption("Categorías ausentes en el entrenamiento: " + ", ".join(sorted(missing)))
    else:
        st.info("La estación carece del historial consecutivo o de las categorías necesarias para ajustar el modelo.")

with chat:
    st.header("Asistente de calidad del aire")
    st.caption(f"Contexto: {station} · {start} a {end} · Referencia predictiva: {reference or 'no disponible'}")
    suggestions = {
        "Estaciones": "¿Qué estaciones están disponibles?",
        "Mediciones": f"¿Cuáles fueron los registros de PM2.5 de la estación «{station}» entre {start} y {end}?",
        "POECA": "¿Qué medidas recomienda el protocolo POECA durante episodios críticos?",
        "Salud": "¿Qué documentos explican los efectos de la contaminación atmosférica en la salud?",
    }
    if reference:
        suggestions["Predicción"] = f"¿Qué categoría predice el modelo para el día siguiente a {reference} en la estación «{station}»?"
    columns = st.columns(len(suggestions))
    suggestion = None
    for col, (label, question_text) in zip(columns, suggestions.items()):
        if col.button(label, key=f"suggest-{label}", disabled=label == "Mediciones" and start > end):
            suggestion = question_text
    typed = st.chat_input("Escribe tu pregunta")
    question = typed or suggestion
    history = st.session_state.setdefault("chat_history", [])
    if question:
        with st.spinner("Consultando el asistente…"):
            answer = ask_agent(
                question, station, reference,
                date_range=(start.isoformat(), end.isoformat()) if start <= end else None,
            )
        history.append((question, answer))
    for prompt, result in history:
        with st.chat_message("user"):
            st.write(prompt)
        with st.chat_message("assistant"):
            if result["modo"] == "llm":
                st.caption("Respuesta generada por el LLM · Groq")
            elif result["modo"] == "respaldo":
                st.caption("Respuesta de respaldo sin LLM")
            else:
                st.caption("Error de herramienta")
            st.markdown(result["respuesta"])
            if result["error_modelo"]:
                st.caption(result["error_modelo"])
            if result["herramientas_usadas"]:
                st.caption("Herramientas: " + ", ".join(result["herramientas_usadas"]))
            if result["fuentes"]:
                with st.expander("Fuentes documentales"):
                    for source in result["fuentes"]:
                        st.write(f"**{source['documento']}** · pág. {source['pagina']}")
                        st.caption(source["contenido"])
