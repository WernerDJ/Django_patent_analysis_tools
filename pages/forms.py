from django import forms

class ExcelUploadForm(forms.Form):
    excel_file = forms.FileField()
    ipc_groups = forms.CharField(
        required=False, 
        label="IPC Groups",
        help_text="Enter one or more IPC groups, separated by commas."
    )
    start_year = forms.IntegerField(
        initial=1900, 
        min_value=1900, 
        max_value=2040,
        label="Start Year"
    )
    end_year = forms.IntegerField(
        initial=2040, 
        min_value=1900, 
        max_value=2040,
        label="End Year"
    )
