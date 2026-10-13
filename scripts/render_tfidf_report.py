"""Genera el PDF de resultados TF-IDF a partir del texto y las tablas."""

from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs/tables"
FIGURES = ROOT / "outputs/figures"
BODY = ParagraphStyle("Body", fontName="Helvetica", fontSize=10.5,
                      leading=15, alignment=TA_JUSTIFY, spaceAfter=9)
LABEL = ParagraphStyle("Label", parent=BODY, alignment=0, keepWithNext=True,
                       spaceBefore=9, spaceAfter=8)
CELL = ParagraphStyle("Cell", fontName="Helvetica", fontSize=8.5, leading=11)
WIDTH = A4[0] - 4 * cm


def table(frame, column_labels=None):
    labels = column_labels or list(frame.columns)
    rows = [[Paragraph(escape(str(label)), CELL) for label in labels]]
    for values in frame.itertuples(index=False, name=None):
        rows.append([Paragraph(escape(f"{value:.4f}" if isinstance(value, float) else str(value)), CELL)
                     for value in values])
    if len(labels) > 2 and str(frame.columns[0]) == "modelo":
        widths = [WIDTH * 0.33] + [WIDTH * 0.67 / (len(labels) - 1)] * (len(labels) - 1)
    else:
        widths = [WIDTH / len(labels)] * len(labels)
    result = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
    result.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eeeeee")),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.HexColor("#999999")),
        ("LINEBELOW", (0, 1), (-1, -1), 0.25, colors.HexColor("#dddddd")),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return result


def metric_tables(filename):
    frame = pd.read_csv(TABLES / filename)
    return [
        table(frame[["modelo", "rmse_content", "rmse_wording", "mcrmse"]],
              ["Modelo", "RMSE content", "RMSE wording", "MCRMSE"]),
        Spacer(1, 10),
        table(frame[["modelo", "mae_content", "mae_wording", "r2_content", "r2_wording"]],
              ["Modelo", "MAE content", "MAE wording", "R2 content", "R2 wording"]),
        Spacer(1, 10),
        table(frame[["modelo", "tiempo_busqueda_s", "tiempo_entrenamiento_s", "tiempo_prediccion_s"]],
              ["Modelo", "Búsqueda s", "Entrenamiento s", "Predicción s"]),
    ]


def footer(canvas, document):
    canvas.saveState()
    canvas.setFont("Helvetica", 9)
    canvas.drawRightString(A4[0] - 2 * cm, 1.25 * cm, str(document.page))
    canvas.restoreState()


def main():
    parts = (ROOT / "docs/resultados_tfidf.txt").read_text(encoding="utf-8").strip().split("\n\n")
    story = []
    skip_next = False
    sections = {"Representación TF-IDF, similitud y resultados de modelos", "Metodología",
                "Resultados de la representación", "Resultados de similitud",
                "Resultados de modelos con TF-IDF y comparación con BoW", "Ajuste de parámetros",
                "Visualizaciones de resultados", "Entrenamiento final y uso con entradas nuevas",
                "Hallazgos y conclusiones", "Referencias"}
    for part in parts:
        if skip_next:
            skip_next = False
            continue
        if part.startswith("Tabla "):
            story.append(Paragraph(escape(part), LABEL))
            number = int(part.split(".")[0].split()[1])
            skip_next = True
            if number == 1:
                frame = pd.read_csv(TABLES / "07_tfidf_vocabulary_diagnostic.csv")
                story.append(table(frame, ["N-gramas", "min_df", "Vocabulario", "Celdas activas", "Dispersión"]))
            elif number == 2:
                frame = pd.read_csv(TABLES / "07_tfidf_term_weights.csv").head(15)
                story.append(table(frame, ["Término", "IDF", "Peso promedio", "Documentos"]))
            elif number == 3:
                frame = pd.read_csv(TABLES / "07_tfidf_similarity_by_prompt.csv")
                story.append(table(frame[["prompt_id", "similitud_media", "similitud_mediana", "spearman_content", "spearman_wording"]],
                                   ["Prompt", "Media", "Mediana", "Spearman content", "Spearman wording"]))
            elif number == 4:
                story.extend(metric_tables("07_tfidf_holdout_results.csv"))
            elif number == 5:
                story.extend(metric_tables("07_tfidf_grouped_results.csv"))
            elif number == 6:
                frame = pd.read_csv(TABLES / "07_tfidf_paired_vs_bow.csv")
                story.append(table(frame.iloc[:, :5]))
                story.append(Spacer(1, 10))
                story.append(table(frame[["modelo", "prompts_con_menor_error", "diferencia_promedio"]],
                                   ["Modelo", "Prompts con menor error", "Diferencia promedio"]))
            elif number == 7:
                frame = pd.read_csv(TABLES / "07_tfidf_selected_parameters.csv")
                for row in frame.itertuples():
                    text = f"{row.protocolo}, fold {row.fold}, {row.modelo}. Parámetros: {row.parametros}. MCRMSE interno: {row.mcrmse_validacion:.4f}."
                    story.append(Paragraph(escape(text), BODY))
            story.append(Spacer(1, 10))
        else:
            story.append(Paragraph(escape(part), LABEL if part in sections else BODY))
        if part == "Visualizaciones de resultados":
            images = [
                ("07_tfidf_top_terms.png", "Figura 1. Términos con mayor peso TF-IDF promedio."),
                ("07_tfidf_similarity_distribution.png", "Figura 2. Distribución de similitud global y por prompt."),
                ("07_tfidf_similarity_vs_targets.png", "Figura 3. Similitud y puntuaciones observadas."),
                ("07_tfidf_model_rmse.png", "Figura 4. RMSE externo de content y wording."),
                ("07_tfidf_protocol_comparison.png", "Figura 5. Error con prompts conocidos y no vistos."),
                ("07_tfidf_time_vs_error.png", "Figura 6. Tiempo de entrenamiento y error externo."),
                ("07_tfidf_prediction_vs_actual.png", "Figura 7. Predicciones externas y puntuaciones reales."),
            ]
            for filename, caption in images:
                image = Image(str(FIGURES / filename))
                scale = min(WIDTH / image.imageWidth, 8 * cm / image.imageHeight)
                image.drawWidth = image.imageWidth * scale
                image.drawHeight = image.imageHeight * scale
                story.append(KeepTogether([image, Spacer(1, 5), Paragraph(escape(caption), BODY), Spacer(1, 9)]))
    target = ROOT / "docs/resultados_tfidf.pdf"
    document = SimpleDocTemplate(str(target), pagesize=A4, rightMargin=2 * cm,
                                leftMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                                title="Representación TF-IDF y resultados de modelos", author="")
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"Informe generado: {target.name}")


if __name__ == "__main__":
    main()
