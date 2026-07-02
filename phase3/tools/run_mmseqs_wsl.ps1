param(
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$MmseqsArgs
)

$script = "/mnt/e/pep/phase3/tools/mmseqs2_local/mmseqs.sh"
wsl.exe bash -lc "$script $($MmseqsArgs -join ' ')"
