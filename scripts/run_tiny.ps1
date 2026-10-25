param(
    [ValidateSet('Test', 'Train', 'Eval')][string]$Mode = 'Test',
    [ValidateSet('rgb', 'll', 'll_lh', 'll_hl', 'll_hh', 'all')][string]$Variant = 'll',
    [string]$Python = (Join-Path $env:USERPROFILE 'anaconda3/envs/dgfnet/python.exe'),
    [string]$McunetRoot = (Join-Path $env:USERPROFILE 'Desktop/Mcunet/mcunet'),
    [int]$BatchSize = 32,
    [int]$Workers = 2,
    [switch]$Resume
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$dataRoot = Join-Path (Split-Path -Parent $projectRoot) 'datasets/tiny-imagenet-200'
$runDirectory = Join-Path $projectRoot "runs/tiny_imagenet/$Variant"
if ($Mode -eq 'Test') {
    & $Python (Join-Path $PSScriptRoot 'test_tiny_pipeline.py') --data-root $dataRoot --mcunet-root $McunetRoot --workers $Workers
} elseif ($Mode -eq 'Train') {
    $trainArguments = @(
        (Join-Path $PSScriptRoot 'train_imagenet.py'),
        '--dataset', 'tiny-imagenet', '--data-root', (Join-Path $dataRoot 'train'),
        '--variant', $Variant, '--run-dir', $runDirectory,
        '--split-manifest', (Join-Path $projectRoot 'splits/tiny_train_dev_seed2026.json'),
        '--config', (Join-Path $projectRoot 'weights/mcunet-256kb-1mb_imagenet.json'),
        '--weights', (Join-Path $projectRoot 'weights/mcunet-256kb-1mb_imagenet.pth'),
        '--mcunet-root', $McunetRoot, '--epochs', '20', '--batch-size', "$BatchSize",
        '--workers', "$Workers", '--seed', '2026', '--lr', '0.001',
        '--weight-decay', '0.00004', '--dev-fraction', '0.05', '--device', 'cuda', '--amp'
    )
    if ($Resume) { $trainArguments += @('--resume', (Join-Path $runDirectory 'last.pt')) }
    & $Python @trainArguments
} else {
    & $Python (Join-Path $PSScriptRoot 'eval_imagenet.py') --dataset tiny-imagenet --data-root (Join-Path $dataRoot 'val') --variant $Variant --checkpoint (Join-Path $runDirectory 'best.pt') --predictions (Join-Path $runDirectory 'val_predictions.csv') --mcunet-root $McunetRoot --batch-size $BatchSize --workers $Workers --device cuda
}
if ($LASTEXITCODE -ne 0) { throw "Tiny ImageNet command failed with exit code $LASTEXITCODE" }
