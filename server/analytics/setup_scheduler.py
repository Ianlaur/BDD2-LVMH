"""
Automated Training Scheduler

Sets up cron jobs or background tasks for:
- Daily incremental updates
- Weekly full retrains
- Monthly performance reviews
"""

from pathlib import Path
import json
from datetime import datetime
import subprocess
import sys


def create_daily_script():
    """Create bash script for daily incremental updates."""
    
    script = """#!/bin/bash
# Daily Incremental ML Update
# Run at 6:00 AM every day

cd /Users/ian/BDD2-LVMH
source .venv/bin/activate

echo "=========================================="
echo "Daily ML Update - $(date)"
echo "=========================================="

# Run incremental update
python -m server.analytics.retrain_workflow << EOF
1
EOF

echo ""
echo "✅ Daily update complete!"
"""
    
    script_path = Path("scripts/ml_daily_update.sh")
    script_path.parent.mkdir(exist_ok=True)
    
    with open(script_path, 'w') as f:
        f.write(script)
        
    # Make executable
    subprocess.run(['chmod', '+x', str(script_path)])
    
    print(f"✓ Created: {script_path}")
    return script_path


def create_weekly_script():
    """Create bash script for weekly full retrain."""
    
    script = """#!/bin/bash
# Weekly Full ML Retrain
# Run at 2:00 AM every Sunday

cd /Users/ian/BDD2-LVMH
source .venv/bin/activate

echo "=========================================="
echo "Weekly ML Retrain - $(date)"
echo "=========================================="

# Run full retrain
python -m server.analytics.retrain_workflow << EOF
2
EOF

echo ""
echo "✅ Weekly retrain complete!"
"""
    
    script_path = Path("scripts/ml_weekly_retrain.sh")
    
    with open(script_path, 'w') as f:
        f.write(script)
        
    # Make executable
    subprocess.run(['chmod', '+x', str(script_path)])
    
    print(f"✓ Created: {script_path}")
    return script_path


def create_monthly_script():
    """Create bash script for monthly A/B testing."""
    
    script = """#!/bin/bash
# Monthly ML A/B Testing
# Run at 3:00 AM on 1st of month

cd /Users/ian/BDD2-LVMH
source .venv/bin/activate

echo "=========================================="
echo "Monthly ML A/B Test - $(date)"
echo "=========================================="

# Run A/B testing
python -m server.analytics.retrain_workflow << EOF
3
EOF

echo ""
echo "✅ Monthly A/B test complete!"
"""
    
    script_path = Path("scripts/ml_monthly_ab_test.sh")
    
    with open(script_path, 'w') as f:
        f.write(script)
        
    # Make executable
    subprocess.run(['chmod', '+x', str(script_path)])
    
    print(f"✓ Created: {script_path}")
    return script_path


def setup_cron_jobs():
    """Generate crontab entries for scheduling."""
    
    print("\n📅 Setting up automated scheduling...\n")
    
    # Create scripts
    daily_script = create_daily_script()
    weekly_script = create_weekly_script()
    monthly_script = create_monthly_script()
    
    # Generate crontab entries
    cron_entries = f"""
# LVMH ML Continuous Training Schedule
# Generated on {datetime.now().isoformat()}

# Daily incremental update at 6:00 AM
0 6 * * * {daily_script.absolute()}

# Weekly full retrain at 2:00 AM every Sunday
0 2 * * 0 {weekly_script.absolute()}

# Monthly A/B test at 3:00 AM on 1st of month
0 3 1 * * {monthly_script.absolute()}
"""
    
    # Save crontab config
    cron_file = Path("scripts/crontab_ml.txt")
    with open(cron_file, 'w') as f:
        f.write(cron_entries)
        
    print(f"\n✓ Created crontab config: {cron_file}")
    
    print("\n" + "="*80)
    print("INSTALLATION INSTRUCTIONS")
    print("="*80)
    
    print(f"""
To activate automated training:

1. Review the crontab entries:
   cat {cron_file}

2. Add to your crontab:
   crontab -e
   
3. Paste the entries from {cron_file}

4. Verify installation:
   crontab -l

5. Check logs:
   tail -f /tmp/cron_ml_*.log


MANUAL TESTING:

Test daily update:
   {daily_script}

Test weekly retrain:
   {weekly_script}

Test monthly A/B:
   {monthly_script}
""")


def create_systemd_services():
    """Create systemd service files (for Linux servers)."""
    
    print("\n🐧 Creating systemd services (Linux)...\n")
    
    # Daily service
    daily_service = """[Unit]
Description=LVMH ML Daily Incremental Update
After=network.target

[Service]
Type=oneshot
User=ian
WorkingDirectory=/Users/ian/BDD2-LVMH
ExecStart=/Users/ian/BDD2-LVMH/scripts/ml_daily_update.sh
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
"""
    
    daily_timer = """[Unit]
Description=LVMH ML Daily Update Timer
Requires=lvmh-ml-daily.service

[Timer]
OnCalendar=daily
OnCalendar=*-*-* 06:00:00
Persistent=true

[Install]
WantedBy=timers.target
"""
    
    service_dir = Path("scripts/systemd")
    service_dir.mkdir(exist_ok=True)
    
    with open(service_dir / "lvmh-ml-daily.service", 'w') as f:
        f.write(daily_service)
        
    with open(service_dir / "lvmh-ml-daily.timer", 'w') as f:
        f.write(daily_timer)
        
    print(f"✓ Created systemd files in: {service_dir}")
    print("""
To install on Linux:
    sudo cp scripts/systemd/* /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable lvmh-ml-daily.timer
    sudo systemctl start lvmh-ml-daily.timer
""")


def create_launchd_plist():
    """Create launchd plist (for macOS)."""
    
    print("\n🍎 Creating launchd plist (macOS)...\n")
    
    plist = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.lvmh.ml.daily</string>
    
    <key>ProgramArguments</key>
    <array>
        <string>/Users/ian/BDD2-LVMH/scripts/ml_daily_update.sh</string>
    </array>
    
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>6</integer>
        <key>Minute</key>
        <integer>0</integer>
    </dict>
    
    <key>StandardOutPath</key>
    <string>/tmp/lvmh_ml_daily.log</string>
    
    <key>StandardErrorPath</key>
    <string>/tmp/lvmh_ml_daily_error.log</string>
    
    <key>WorkingDirectory</key>
    <string>/Users/ian/BDD2-LVMH</string>
</dict>
</plist>
"""
    
    plist_dir = Path("scripts/launchd")
    plist_dir.mkdir(exist_ok=True)
    
    plist_path = plist_dir / "com.lvmh.ml.daily.plist"
    with open(plist_path, 'w') as f:
        f.write(plist)
        
    print(f"✓ Created launchd plist: {plist_path}")
    print(f"""
To install on macOS:
    cp {plist_path} ~/Library/LaunchAgents/
    launchctl load ~/Library/LaunchAgents/com.lvmh.ml.daily.plist
    launchctl start com.lvmh.ml.daily

To check status:
    launchctl list | grep lvmh
    
To view logs:
    tail -f /tmp/lvmh_ml_daily.log
""")


def main():
    """Setup automated training."""
    
    print("="*80)
    print("ML CONTINUOUS TRAINING - AUTOMATED SCHEDULER")
    print("="*80)
    
    print("\nThis will set up automated training schedules:")
    print("  • Daily: Incremental updates (6:00 AM)")
    print("  • Weekly: Full retrain (Sunday 2:00 AM)")
    print("  • Monthly: A/B testing (1st at 3:00 AM)")
    
    choice = input("\nProceed? (y/n): ").strip().lower()
    
    if choice != 'y':
        print("Cancelled.")
        return
        
    # Create scripts
    setup_cron_jobs()
    
    # Platform-specific
    if sys.platform == 'darwin':
        create_launchd_plist()
    elif sys.platform == 'linux':
        create_systemd_services()
        
    print("\n✅ Automated training scheduler setup complete!")
    print("\nNext steps:")
    print("  1. Test scripts manually first")
    print("  2. Review and adjust schedules")
    print("  3. Install cron/systemd/launchd jobs")
    print("  4. Monitor logs regularly")


if __name__ == "__main__":
    main()
