<#
.SYNOPSIS
    Push the issue drafts under .github/ISSUE_DRAFTS/ to GitHub via gh CLI.

.DESCRIPTION
    Re-runs are idempotent: any issue whose exact title already exists on the
    remote is skipped. Requires gh (>= 2.30), and an authenticated session
    (gh auth login).

.EXAMPLE
    pwsh scripts/seed_issues.ps1
    pwsh scripts/seed_issues.ps1 -Files .github/ISSUE_DRAFTS/open/06-*.md
#>
param(
    [string[]] $Files
)

$ErrorActionPreference = 'Stop'
$root       = (Resolve-Path "$PSScriptRoot\..").Path
$draftsDir  = Join-Path $root '.github/ISSUE_DRAFTS'

# ---------- prerequisites ---------------------------------------------------
if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
    Write-Error "gh CLI not found"; exit 1
}
gh auth status *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Error "Run 'gh auth login' first"; exit 1
}

# ---------- helpers ---------------------------------------------------------
function Get-Frontmatter {
    param([string] $Path)
    $raw   = Get-Content -LiteralPath $Path -Raw
    if ($raw -notmatch '(?s)^---\s*\r?\n(?<fm>.*?)\r?\n---\s*\r?\n(?<body>.*)$') {
        throw "No YAML frontmatter found in $Path"
    }
    $fmText = $Matches['fm']
    $body   = $Matches['body']

    $fm = @{}
    foreach ($line in $fmText -split "`n") {
        if ($line -match '^\s*([A-Za-z_]+):\s*(.*?)\s*$') {
            $k = $Matches[1]; $v = $Matches[2]
            if ($v -match '^\[(.*)\]$') {
                $items = $Matches[1] -split ',\s*' | ForEach-Object { $_.Trim('"').Trim("'").Trim() } | Where-Object { $_ }
                $fm[$k] = $items
            } else {
                $fm[$k] = $v.Trim('"').Trim("'")
            }
        }
    }
    return [pscustomobject]@{ Frontmatter = $fm; Body = $body }
}

function Test-IssueExists {
    param([string] $Title)
    $json = gh issue list --state all --search "in:title `"$Title`"" --json title | ConvertFrom-Json
    return [bool] ($json | Where-Object { $_.title -eq $Title })
}

function Ensure-Milestone {
    param([string] $Title)
    if (-not $Title) { return }
    $existing = gh api 'repos/{owner}/{repo}/milestones' --paginate | ConvertFrom-Json
    if (-not ($existing | Where-Object { $_.title -eq $Title })) {
        Write-Host "  -> creating milestone $Title"
        gh api -X POST 'repos/{owner}/{repo}/milestones' -f "title=$Title" *> $null
    }
}

# ---------- main ------------------------------------------------------------
if (-not $Files -or $Files.Count -eq 0) {
    $Files = @()
    $Files += (Get-ChildItem -Path (Join-Path $draftsDir 'closed') -Filter *.md).FullName
    $Files += (Get-ChildItem -Path (Join-Path $draftsDir 'open')   -Filter *.md).FullName
}

foreach ($f in $Files) {
    if (-not (Test-Path $f)) { continue }
    $parsed   = Get-Frontmatter -Path $f
    $fm       = $parsed.Frontmatter
    $title    = $fm['title']
    $state    = $fm['state']
    $milestone = $fm['milestone']
    $closedBy = $fm['closed_by']
    $labels   = if ($fm['labels'] -is [array]) { ($fm['labels'] -join ',') } else { $fm['labels'] }

    Write-Host "-> $title"
    if (Test-IssueExists -Title $title) {
        Write-Host "  -> already exists on remote, skipping"
        continue
    }

    Ensure-Milestone -Title $milestone

    $ghArgs = @('issue', 'create', '--title', $title, '--body', $parsed.Body)
    if ($labels)    { $ghArgs += @('--label',     $labels) }
    if ($milestone) { $ghArgs += @('--milestone', $milestone) }

    $url = (& gh @ghArgs).Trim()
    Write-Host "  -> created $url"

    if ($state -eq 'closed') {
        $comment = "Closed by **$closedBy** -- shipped in v1.0."
        & gh issue close $url --comment $comment *> $null
        Write-Host "  -> closed ($closedBy)"
    }
}

Write-Host "Done."
