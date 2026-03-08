param(
    [string]$RepoName = "philosophy-debate-ai",
    [ValidateSet("public", "private")]
    [string]$Visibility = "public",
    [string]$Owner = "",
    [string]$RemoteName = "origin",
    [string]$Branch = "main",
    [string]$CommitMessage = "Initial commit",
    [string]$GitUserName = "Priyanshu Garg",
    [string]$GitUserEmail = "",
    [switch]$SkipRepoCreate
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Require-Command {
    param([string]$Name, [string]$InstallHint)

    $command = Get-Command $Name -ErrorAction SilentlyContinue
    if ($null -eq $command) {
        throw "$Name is not installed or not on PATH. $InstallHint"
    }
}

function Run-Step {
    param([string]$Message, [scriptblock]$Action)

    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
    & $Action
}

function Invoke-External {
    param(
        [Parameter(Mandatory = $true)]
        [string]$FilePath,
        [string[]]$Arguments = @(),
        [switch]$AllowFailure
    )

    & $FilePath @Arguments
    $exitCode = $LASTEXITCODE
    if (-not $AllowFailure -and $exitCode -ne 0) {
        $joined = if ($Arguments.Count -gt 0) { "$FilePath " + ($Arguments -join " ") } else { $FilePath }
        throw "Command failed with exit code $exitCode: $joined"
    }
    return $exitCode
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

Require-Command -Name "git" -InstallHint "Install Git for Windows first: https://git-scm.com/download/win"
Require-Command -Name "gh" -InstallHint "Install GitHub CLI and run 'gh auth login' first: https://cli.github.com/"

Run-Step -Message "Checking GitHub authentication" -Action {
    Invoke-External -FilePath "gh" -Arguments @("auth", "status")
}

$trackedPaths = @(
    ".gitignore",
    ".env.example",
    "app.py",
    "README.md",
    "WINDOWS_SETUP.md",
    "Runtime Instructions.txt",
    "requirements.txt",
    "packages.txt",
    ".streamlit",
    "philosophy_debate",
    "scripts",
    "Stoicism Corpus",
    "Vedanta corpus",
    "Machiavellianism Corpus",
    "storage"
)

if (-not (Test-Path ".git")) {
    Run-Step -Message "Initializing local Git repository" -Action {
        Invoke-External -FilePath "git" -Arguments @("init")
    }
}

Run-Step -Message "Configuring branch name" -Action {
    Invoke-External -FilePath "git" -Arguments @("branch", "-M", $Branch)
}

$currentUserName = (git config --get user.name) 2>$null
$currentUserEmail = (git config --get user.email) 2>$null

if ([string]::IsNullOrWhiteSpace($currentUserName)) {
    Run-Step -Message "Setting local Git user.name" -Action {
        Invoke-External -FilePath "git" -Arguments @("config", "user.name", $GitUserName)
    }
}

if ([string]::IsNullOrWhiteSpace($currentUserEmail)) {
    if ([string]::IsNullOrWhiteSpace($GitUserEmail)) {
        throw "Git user.email is not configured. Re-run the script with -GitUserEmail 'your_email@example.com' or set it with: git config user.email 'your_email@example.com'"
    }
    Run-Step -Message "Setting local Git user.email" -Action {
        Invoke-External -FilePath "git" -Arguments @("config", "user.email", $GitUserEmail)
    }
}

Run-Step -Message "Staging publishable project files" -Action {
    Invoke-External -FilePath "git" -Arguments (@("add", "--") + $trackedPaths)
}

Invoke-External -FilePath "git" -Arguments @("rev-parse", "--verify", "HEAD") -AllowFailure | Out-Null
$hasCommits = $LASTEXITCODE -eq 0

$statusOutput = git status --short
if (-not $hasCommits -or -not [string]::IsNullOrWhiteSpace(($statusOutput | Out-String))) {
    Run-Step -Message "Creating commit" -Action {
        Invoke-External -FilePath "git" -Arguments @("commit", "-m", $CommitMessage)
    }
} else {
    Write-Host ""
    Write-Host "==> No new local changes to commit" -ForegroundColor Yellow
}

Invoke-External -FilePath "git" -Arguments @("remote", "get-url", $RemoteName) -AllowFailure | Out-Null
$remoteExists = $LASTEXITCODE -eq 0

$repoSlug = if ([string]::IsNullOrWhiteSpace($Owner)) { $RepoName } else { "$Owner/$RepoName" }
$defaultDescription = "A multi-agent RAG philosophy debate app built with Streamlit, LangChain, Groq, and Chroma."

if (-not $SkipRepoCreate -and -not $remoteExists) {
    Run-Step -Message "Creating GitHub repository $repoSlug" -Action {
        Invoke-External -FilePath "gh" -Arguments @(
            "repo", "create", $repoSlug,
            "--$Visibility",
            "--source", ".",
            "--remote", $RemoteName,
            "--description", $defaultDescription,
            "--push"
        )
    }
} else {
    if (-not $remoteExists) {
        throw "No remote named '$RemoteName' exists. Remove -SkipRepoCreate or add the remote first."
    }

    Run-Step -Message "Pushing branch $Branch to $RemoteName" -Action {
        Invoke-External -FilePath "git" -Arguments @("push", "-u", $RemoteName, $Branch)
    }
}

Run-Step -Message "Done" -Action {
    Invoke-External -FilePath "gh" -Arguments @("repo", "view", $repoSlug, "--json", "url", "--jq", ".url")
}