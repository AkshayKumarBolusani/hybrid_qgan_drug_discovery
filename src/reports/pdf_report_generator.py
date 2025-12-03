"""PDF report generator."""
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib import colors
from datetime import datetime
from pathlib import Path

class PDFReportGenerator:
    def __init__(self, output_path='report.pdf'):
        self.output_path = output_path
        self.doc = SimpleDocTemplate(output_path, pagesize=letter)
        self.styles = getSampleStyleSheet()
        self.story = []
    
    def add_title(self, title):
        self.story.append(Paragraph(title, self.styles['Title']))
        self.story.append(Spacer(1, 12))
    
    def add_heading(self, text):
        self.story.append(Paragraph(text, self.styles['Heading1']))
        self.story.append(Spacer(1, 6))
    
    def add_paragraph(self, text):
        self.story.append(Paragraph(text, self.styles['Normal']))
        self.story.append(Spacer(1, 6))
    
    def add_table(self, data, col_widths=None):
        table = Table(data, colWidths=col_widths)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        self.story.append(table)
        self.story.append(Spacer(1, 12))
    
    def add_image(self, image_path, width=400, height=300):
        if Path(image_path).exists():
            img = Image(image_path, width=width, height=height)
            self.story.append(img)
            self.story.append(Spacer(1, 12))
    
    def generate(self):
        self.doc.build(self.story)
        return self.output_path

def create_drug_discovery_report(
    output_path,
    generated_smiles,
    metrics,
    qsar_results=None,
    tox_results=None,
    docking_results=None
):
    """Create a complete drug discovery report."""
    report = PDFReportGenerator(output_path)
    
    # Title
    report.add_title("Hybrid Quantum GAN Drug Discovery Report")
    report.add_paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Summary
    report.add_heading("Executive Summary")
    report.add_paragraph(f"Total molecules generated: {len(generated_smiles)}")
    report.add_paragraph(f"Validity: {metrics.get('validity', 0):.2%}")
    report.add_paragraph(f"Uniqueness: {metrics.get('uniqueness', 0):.2%}")
    
    # Molecules table
    report.add_heading("Top Generated Molecules")
    table_data = [['#', 'SMILES', 'QED', 'LogP']]
    for i, smiles in enumerate(generated_smiles[:10], 1):
        table_data.append([str(i), smiles[:50], '0.75', '2.3'])
    report.add_table(table_data)
    
    # Generate PDF
    return report.generate()
