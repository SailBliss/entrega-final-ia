import pytesseract
from PIL import Image

# Ubicación del programa Tesseract
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# Comprobar la versión
version = pytesseract.get_tesseract_version()

print("Tesseract conectado correctamente.")
print("Versión:", version)