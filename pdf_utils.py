# pdf_utils.py
import os
from flask import render_template, url_for, current_app
from weasyprint import HTML

def generate_pdf(result, petNo):
    html = render_template(
      'templats_pdf.html',
      disease            = result.get('disease',''),
      prescription_plan  = result['prescription_plan'],
      diet_plan          = result['diet_plan'],
      explanation        = result['explanation']
    )
    pdf_bytes = HTML(string=html).write_pdf()
    filename   = f"{petNo}.pdf"
    path       = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
    with open(path, 'wb') as f:
        f.write(pdf_bytes)
    return url_for('static', filename=f'uploads/{filename}', _external=True)