$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

# Initialize log early so we can capture branch decisions
$logPath = Join-Path $root 'build_exe.log'
Set-Content -Path $logPath -Value "[start] $(Get-Date -Format o)" -Encoding utf8

function Write-Log {
  param(
    [Parameter(Mandatory=$true)][string]$Message
  )
  Write-Host $Message
  Add-Content -Path $logPath -Value $Message
}

# Ensure native command stderr doesn't promote to terminating errors
$global:PSNativeCommandUseErrorActionPreference = $false

# Resolve Python interpreter (prefer repo venv)
$venvPy = Join-Path $root 'scaffold_env_312_py31210\Scripts\python.exe'
if (Test-Path $venvPy) {
  $python = $venvPy
} else {
  $python = 'python'
}

# Pip occasionally writes warnings to stderr; avoid treating as fatal
$oldEap = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
try {
  & $python -m pip install --upgrade pip *> $null
  & $python -m pip install pyinstaller python-dotenv --quiet *> $null
} finally {
  $ErrorActionPreference = $oldEap
}

# CPU (default) vs GPU build
$pyiExcludes = @('--exclude-module','torch.cuda','--exclude-module','torch.backends.cudnn','--exclude-module','torch.backends.cuda','--exclude-module','torch.backends.cublas')
if ($env:SC_INCLUDE_CUDA -eq '1') {
  Write-Log '[i] GPU build requested via SC_INCLUDE_CUDA=1'
  $pyiExcludes = @()
  Write-Log '[i] Installing CUDA-enabled libraries (torch/cu121 and onnxruntime-gpu)'
  $oldEap = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
  try {
    & $python -m pip uninstall -y onnxruntime *> $null
    & $python -m pip install --upgrade onnxruntime-gpu --quiet *> $null
    & $python -m pip uninstall -y torch torchvision torchaudio *> $null
    & $python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio --quiet *> $null
  } finally { $ErrorActionPreference = $oldEap }
} else {
  Write-Log '[i] CPU-only build (default). To try GPU, set SC_INCLUDE_CUDA=1'
  $oldEap = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
  try {
    & $python -m pip uninstall -y onnxruntime-gpu *> $null
    & $python -m pip install --upgrade onnxruntime --quiet *> $null
    & $python -m pip uninstall -y torch torchvision torchaudio *> $null
    & $python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio --quiet *> $null
  } finally { $ErrorActionPreference = $oldEap }
}

$args = @(
  '--noconfirm','--clean','--onedir','--name','ScaffoldAI-EnhancedUI',
  '--hidden-import','encodings',
  '--hidden-import','codecs',
  '--hidden-import','scipy.special._cdflib'
) + $pyiExcludes + @(
  '--add-data','frontend;frontend',
  '--add-data','scaffold_core;scaffold_core',
  '--add-data','model_config.json;.',
  '--add-data','vector_outputs\scaffold_index_1.faiss;vector_outputs',
  '--add-data','vector_outputs\scaffold_metadata_1.json;vector_outputs',
  '--collect-all','torch',
  '--collect-all','torchvision',
  '--collect-all','torchaudio',
  '--collect-all','sentence_transformers',
  '--collect-all','transformers',
  '--collect-all','faiss',
  '--collect-all','psutil',
  '--collect-all','flask','--collect-all','flask_cors',
  '--collect-all','werkzeug',
  '--collect-all','jinja2',
  '--collect-all','optimum',
  # Prefer GPU ONNX Runtime; ensure import symbol is present
  '--collect-all','onnxruntime_gpu',
  '--hidden-import','onnxruntime',
  '--collect-all','fitz',
  '--collect-all','pymupdf',
  '--collect-all','PyPDF2',
  '--collect-all','pypdf',
  '--collect-all','pdfminer',
  '--collect-all','pdfplumber',
  'run_enhanced_ui.py'
)

Write-Log '[i] Starting PyInstaller...'
$oldEap = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
$oldNative = $PSNativeCommandUseErrorActionPreference
$PSNativeCommandUseErrorActionPreference = $false
try {
  & $python -m PyInstaller @args 2>&1 | Tee-Object -FilePath $logPath -Append
} finally {
  $ErrorActionPreference = $oldEap
  $PSNativeCommandUseErrorActionPreference = $oldNative
}

if (Test-Path (Join-Path $root 'dist\ScaffoldAI-EnhancedUI')) {
  if (Test-Path (Join-Path $root '.env')) {
    Copy-Item (Join-Path $root '.env') (Join-Path $root 'dist\ScaffoldAI-EnhancedUI\.env') -Force
  }
  if (Test-Path (Join-Path $root '.env.local')) {
    Copy-Item (Join-Path $root '.env.local') (Join-Path $root 'dist\ScaffoldAI-EnhancedUI\.env.local') -Force
  }
}

Write-Log "Build complete. See $logPath"


