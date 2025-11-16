import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from flask import send_file  # or FastAPI: StreamingResponse

def generate_pdf_from_text(text, title="Document"):
    """
    Generate a PDF from a string of text and return it as a BytesIO object.
    """
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    PAGE_WIDTH, PAGE_HEIGHT = letter
    LEFT_MARGIN = 50
    TOP_MARGIN = 750
    LINE_HEIGHT = 20
    FONT_SIZE = 12
    FONT_NAME = "Helvetica"

    c.setFont(FONT_NAME, FONT_SIZE)
    
    y = TOP_MARGIN
    # Add title
    c.drawString(LEFT_MARGIN, y, title)
    y -= LINE_HEIGHT * 2

    for line in text.split("\n"):
        if y < 50:  # bottom margin, new page
            c.showPage()
            c.setFont(FONT_NAME, FONT_SIZE)
            y = TOP_MARGIN
        c.drawString(LEFT_MARGIN, y, line)
        y -= LINE_HEIGHT

    c.save()
    buffer.seek(0)
    return buffer

# -----------------------------
# Example usage with Flask
# -----------------------------
# from flask import Flask, request
# app = Flask(__name__)
#
# @app.route("/get_pdf")
# def get_pdf():
#     # Replace this with text from your RAG or file
#     text = "Data Science Curriculum:\n1. Python Basics\n2. Data Analysis\n..."
#     pdf_buffer = generate_pdf_from_text(text, title="Data Science Curriculum")
#     return send_file(pdf_buffer, as_attachment=True, download_name="curriculum.pdf", mimetype="application/pdf")
#
# if __name__ == "__main__":
#     app.run(debug=True)
