# Khoj-o-Drone

Autonomous disaster-zone search & rescue drone simulation (Cyberwave SDK + Python).

## Setup (Windows PowerShell)

```powershell
py -3.14 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install cyberwave numpy matplotlib
```

## Run offline (no Cyberwave needed)

```powershell
python khoj_o_drone.py --seed 7
```

Writes `khoj_output\mission_report_seed7.json` and `khoj_output\mission_map_seed7.png`.

## Run in Cyberwave

```powershell
$env:CYBERWAVE_API_KEY = "your_api_key"
python khoj_o_drone.py --cyberwave --twin-id "YOUR_DRONE_TWIN_ID" --speed 4 --origin -10 -10 0
```

Open your environment page in the Cyberwave dashboard and refresh it once the terminal prints `scene ready`.

## Other commands

```powershell
python khoj_o_drone.py --runs 10 --seed 100 --no-plot   # 10 seeded missions + pass rate
python khoj_o_drone.py --cyberwave --twin-id "YOUR_DRONE_TWIN_ID" --clear-scene   # remove old scene objects
python khoj_o_drone.py --help                            # all options
```
