# Documentación

Esta carpeta contendrá los entregables finales del proyecto:

- informe de análisis exploratorio en PDF;
- presentación de resultados;
- material complementario necesario para documentar decisiones del análisis.

Los resultados tabulares y gráficos reproducibles permanecen en `outputs/`.

La sección resultados_tfidf.txt contiene la metodología, las métricas,
el análisis de similitud, la comparación con BoW y las conclusiones de
TF-IDF en texto sencillo. Se genera al ejecutar 07_tfidf_analysis.ipynb.
La versión PDF con las tablas y figuras se genera con
scripts/render_tfidf_report.py y se conserva en resultados_tfidf.pdf.
Los resultados se separan según la evaluación de prompts conocidos o de
prompts no vistos. Las cifras del informe se obtienen de las mismas tablas
que se exportan en outputs/tables.

Los informes informe_eda_commonlit.docx e informe_eda_commonlit.pdf
conservan el análisis exploratorio. La sección TF-IDF y sus figuras se
utilizan para completar el informe general de resultados.
