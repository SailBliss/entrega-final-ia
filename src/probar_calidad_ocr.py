from src.rag import CARPETA_PDF, leer_pdf


documentos = [
    "acuerdo_16_2017_pigeca_valle_aburra.pdf",
    "resolucion_2254_2017_norma_calidad_aire.pdf"
]

carpeta = CARPETA_PDF

for nombre in documentos:

    ruta = carpeta / nombre

    print("\n" + "=" * 70)
    print("DOCUMENTO:", nombre)

    paginas = leer_pdf(ruta)

    for pagina in paginas:

        print("\nPÁGINA:", pagina["pagina"])
        print("CARACTERES:", len(pagina["texto"]))
        print("-" * 50)
        print(pagina["texto"][:1500])
