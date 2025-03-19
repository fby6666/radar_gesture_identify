@echo off
setlocal enabledelayedexpansion

set "count=0"
for %%f in (*.jpg) do (
    set /a "count+=1"
    set "newname=0001"
    set "newname=!newname:~0,-%count%!!count!"
    if !count! leq 20 (
        ren "%%f" "!newname!.jpg"
    )
)

echo Renaming completed.
pause