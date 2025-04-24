from django import forms

class SimpleExcelUploadForm(forms.Form):
    excel_file = forms.FileField(label="Upload Excel File")
    )
