@echo off
setlocal
cd /d %~dp0

REM Build the LaTeX report and output PDF to report\ using MiKTeX.
if not exist report mkdir report

set TEXSRC=report\scaffold_ai_status_report.tex
set PDFOUT=report\scaffold_ai_status_report.pdf

where latexmk >nul 2>&1
if %ERRORLEVEL%==0 (
  echo Using latexmk...
  latexmk -pdf -halt-on-error -interaction=nonstopmode %TEXSRC%
  if %ERRORLEVEL% NEQ 0 (
    echo latexmk failed, falling back to pdflatex...
    goto :pdflatex
  )
  if not exist %PDFOUT% (
    echo latexmk did not produce PDF, falling back to pdflatex...
    goto :pdflatex
  )
  goto :done
) else (
  echo latexmk not found. Falling back to pdflatex (MiKTeX)...
  goto :pdflatex
)

:
:pdflatex
pdflatex -halt-on-error -interaction=nonstopmode -output-directory=report %TEXSRC%
if %ERRORLEVEL% NEQ 0 goto :fail
REM Run bibtex only if a .aux exists with citations (not used here)
if exist report\scaffold_ai_status_report.aux (
  findstr /i "citation" report\scaffold_ai_status_report.aux >nul 2>&1 && (
    echo Running bibtex...
    pushd report
    bibtex scaffold_ai_status_report.aux
    popd
  )
)
pdflatex -halt-on-error -interaction=nonstopmode -output-directory=report %TEXSRC%
pdflatex -halt-on-error -interaction=nonstopmode -output-directory=report %TEXSRC%
if not exist %PDFOUT% goto :fail
goto :done

:done
echo Done. Output: %PDFOUT%
exit /b 0

:fail
echo Build failed. Ensure MiKTeX is installed and pdflatex is on PATH.
exit /b 1


