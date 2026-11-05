# Renders docs/demo.gif. Run from the repo root on Windows: vhs docs/demo.tape
# Needs: vhs (with ttyd, ffmpeg), Ollama running with the model in config.yaml.
# PowerShell, not bash: on Windows vhs resolves "bash" to WSL, which has no uv.

Output docs/demo.gif

Set Shell "powershell"
Set FontSize 18
Set Width 1200
Set Height 640
Set Padding 20
Set TypingSpeed 40ms

# Build the sample bots off camera so the recording shows only answers.
Hide
Type "Copy-Item examples\* data\; uv run obot build 2>$null; Clear-Host"
Enter
Wait@10m
Show

Type "# 검색 질문: 근거 레코드를 함께 돌려줍니다"
Enter
Type 'uv run obot ask products "무선 마우스 추천해줘" 2>$null'
Enter
Wait+Screen@3m /sources:/
Sleep 4s

Type "# 집계 질문: SQL로 바꿔 실행합니다"
Enter
Type 'uv run obot ask orders "주문이 총 몇 건이야?" 2>$null'
Enter
Wait+Screen@3m /sql:/
Sleep 5s
