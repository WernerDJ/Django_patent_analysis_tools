from django import forms

class ExcelUploadForm(forms.Form):
    excel_file = forms.FileField()
    ipc_groups = forms.CharField(
        required=False, 
        label="IPC Groups",
        help_text="Enter one or more IPC groups, separated by commas."
    )
    start_year = forms.IntegerField(
        initial=1990, 
        min_value=1990, 
        max_value=2030,
        label="Start Year"
    )
    end_year = forms.IntegerField(
        initial=2030, 
        min_value=1990, 
        max_value=2030,
        label="End Year"
    )

class ReducedExcelUploadForm(forms.Form):
    excel_file = forms.FileField(label="Upload Excel File")
    start_year = forms.IntegerField(
        initial=2005, 
        min_value=2005, 
        max_value=2030,
        label="Start Year"
    )
    end_year = forms.IntegerField(
        initial=2026, 
        min_value=2005, 
        max_value=2030,
        label="End Year"
    )

class SimpleExcelUploadForm(forms.Form):
    excel_file = forms.FileField(label="Upload Excel File")

