Set fso = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")
shell.CurrentDirectory = fso.GetParentFolderName(WScript.ScriptFullName)
If Not fso.FileExists(".venv\Scripts\pythonw.exe") Then
    shell.Run "uv sync", 0, True
End If
shell.Run """.venv\Scripts\pythonw.exe"" -m upscaler", 0, False
