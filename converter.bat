@echo off
cd /d %~dp0

setlocal enabledelayedexpansion
set OUT_DIR=%~dp0doc\output
if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"
echo Select output format:
echo A) Convert .typ to .docx (Word)
echo B) Convert .typ to .tex (LaTeX)
echo C) Convert .typ to .html (HTML)
echo D) Produce all (docx + tex + html)

echo Please enter your choice (A/B/C/D):
echo.
set /p CHOICE=

echo selected %CHOICE%

set OUT_FMT=
set EXT=

if /I "%CHOICE%"=="A" (
    echo Converting to docx
    set "OUT_FMT=docx"
    set "EXT=docx"
    goto :CHOICE_DONE
)

if /I "%CHOICE%"=="B" (
    echo Converting to latex
    set "OUT_FMT=latex"
    set "EXT=tex"
    goto :CHOICE_DONE
)

if /I "%CHOICE%"=="C" (
    echo Converting to html
    set "OUT_FMT=html"
    set "EXT=html"
    goto :CHOICE_DONE
)

if /I "%CHOICE%"=="D" (
    echo Converting to all formats
    set "OUT_FMT=all"
    goto :CHOICE_DONE
)

echo Invalid choice %CHOICE%, Defaulting to A (docx)
set "OUT_FMT=docx"
set "EXT=docx"

:CHOICE_DONE

echo Converting .typ files in doc\snippet to %OUT_FMT%

for %%F in (doc\snippet\*.typ) do (
	echo Preparing %%~nxF
	@REM create a temporary preprocessed copy with chevron to angle replacement
	set TMPFILE=%OUT_DIR%\%%~nF.preprocessed.typ
	powershell -NoProfile -Command "(Get-Content -Raw -Encoding UTF8 -LiteralPath '%%~fF') -replace 'chevron','angle' | Set-Content -Encoding UTF8 -LiteralPath '!TMPFILE!'"
	if exist "!TMPFILE!" (
		if /I "%OUT_FMT%"=="all" (
			echo Creating %OUT_DIR%\%%~nF.docx
			pandoc "!TMPFILE!" -o "%OUT_DIR%\%%~nF.docx"
			echo Creating %OUT_DIR%\%%~nF.tex
			pandoc "!TMPFILE!" -o "%OUT_DIR%\%%~nF.tex"
			echo Creating %OUT_DIR%\%%~nF.html
			pandoc "!TMPFILE!" -o "%OUT_DIR%\%%~nF.html"
		) else (
			echo Creating %OUT_DIR%\%%~nF.!EXT!
			if /I "%OUT_FMT%"=="latex" (
				pandoc "!TMPFILE!" -s -o "%OUT_DIR%\%%~nF.!EXT!"
			) else (
				pandoc "!TMPFILE!" -o "%OUT_DIR%\%%~nF.!EXT!"
			)
		)
		del "!TMPFILE!"
	) else (
		echo [WARN] Preprocessed file not created for %%~nxF, falling back to original
		if /I "%OUT_FMT%"=="all" (
			pandoc "%%~fF" -o "%OUT_DIR%\%%~nF.docx"
			pandoc "%%~fF" -o "%OUT_DIR%\%%~nF.tex"
			pandoc "%%~fF" -o "%OUT_DIR%\%%~nF.html"
		) else if /I "%OUT_FMT%"=="latex" (
			pandoc "%%~fF" -s -o "%OUT_DIR%\%%~nF.!EXT!"
		) else (
			pandoc "%%~fF" -o "%OUT_DIR%\%%~nF.!EXT!"
		)
	)
)

endlocal
