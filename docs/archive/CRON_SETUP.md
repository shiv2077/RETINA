# Nightly Retrain Automation - Cron Setup Guide

## Overview

This guide walks through setting up the `scripts/nightly_retrain.py` script to run automatically every night, closing the active learning loop automatically.

## Prerequisites

- ✅ Script exists: `scripts/nightly_retrain.py`
- ✅ Annotations stored in: `annotations/annotations.json`
- ✅ Base dataset available (Decospan, MVTec AD, or similar)
- ✅ Python environment configured with PyTorch
- ✅ Project root: `/home/shiv2077/dev/RETINA`

## Step-by-Step Setup

### 1. Create Logs Directory

```bash
mkdir -p /home/shiv2077/dev/RETINA/logs
```

This directory will store daily retrain logs for monitoring/debugging.

### 2. Make Script Executable

```bash
chmod +x /home/shiv2077/dev/RETINA/scripts/nightly_retrain.py
```

### 3. Test Script Manually (IMPORTANT)

Before adding to cron, test the script manually in your actual Python environment:

```bash
cd /home/shiv2077/dev/RETINA
python scripts/nightly_retrain.py
```

Watch for:
- ✅ Datasets loaded successfully
- ✅ Model loaded or created
- ✅ Fine-tuning completes without errors
- ✅ Model deployed to `output/bgad_production.pt`
- ✅ Log file created in `logs/`

### 4. Determine Your Python Executable Path

```bash
which python3
# Output will be something like: /usr/bin/python3
```

Or if using conda/venv:

```bash
# For conda
conda info --base
# For venv
which python

```

**NOTE THE PATH - you'll need it for the cron job.**

### 5. Add Cron Job

Open crontab editor:

```bash
crontab -e
```

Add this line (change executable path if needed):

```cron
# Run nightly retrain every day at 2:00 AM
0 2 * * * cd /home/shiv2077/dev/RETINA && /usr/bin/python3 scripts/nightly_retrain.py >> logs/retrain.log 2>&1
```

**Explanation:**
- `0 2 * * *` = Every day at 02:00 (2:00 AM)
- `cd /home/shiv2077/dev/RETINA` = Change to project directory
- `/usr/bin/python3 scripts/nightly_retrain.py` = Run the script
- `>> logs/retrain.log 2>&1` = Append stdout and stderr to log file

### 6. Verify Cron Job Was Added

```bash
crontab -l
```

You should see your cron job listed.

## Alternative Cron Schedules

**Every day at 3:00 AM:**
```cron
0 3 * * * cd /home/shiv2077/dev/RETINA && /usr/bin/python3 scripts/nightly_retrain.py >> logs/retrain.log 2>&1
```

**Every Sunday at 2:00 AM (weekly):**
```cron
0 2 * * 0 cd /home/shiv2077/dev/RETINA && /usr/bin/python3 scripts/nightly_retrain.py >> logs/retrain.log 2>&1
```

**Multiple times daily (every 6 hours):**
```cron
0 */6 * * * cd /home/shiv2077/dev/RETINA && /usr/bin/python3 scripts/nightly_retrain.py >> logs/retrain.log 2>&1
```

## Monitoring & Debugging

### Check Latest Log

```bash
tail -f /home/shiv2077/dev/RETINA/logs/retrain.log
```

### View Last 50 Lines

```bash
tail -50 /home/shiv2077/dev/RETINA/logs/retrain.log
```

### Check All Logs

```bash
ls -lh /home/shiv2077/dev/RETINA/logs/
```

### Parse Success/Failure

```bash
# Check for completion marker
grep "NIGHTLY RETRAIN COMPLETE" /home/shiv2077/dev/RETINA/logs/retrain.log

# Check for failures
grep "RETRAIN FAILED" /home/shiv2077/dev/RETINA/logs/retrain.log
```

### Check Cron Execution

On many Linux systems, cron logs are in:

```bash
journalctl -u cron --since today
# or
sudo tail -f /var/log/cron
```

Check if your job executed:
```bash
grep "nightly_retrain.py" /var/log/cron
```

## Expected Output

After a successful cron execution, `logs/retrain_YYYYMMDD_HHMMSS.log` should contain:

```
================================================================================
RETINA NIGHTLY RETRAIN PIPELINE STARTED
================================================================================
Device: cuda

📦 STAGE 1: Loading datasets...
Loaded 500 normal training images from /home/shiv2077/dev/RETINA/decospan_small/train/good
Found 12 new anomaly annotations ready for retraining

🔀 STAGE 2: Creating combined dataloader...
Sampling 300 normal images (ratio 25:1)
Total training samples: 312

⚙️  STAGE 3: Fine-tuning BGAD model...
Loading existing model from /home/shiv2077/dev/RETINA/output/bgad_production.pt
✅ Successfully loaded production model
Epoch 1/10 | Loss: 1.234 | Pull: 0.456 | Push: 0.778
Epoch 2/10 | Loss: 1.100 | Pull: 0.389 | Push: 0.711
...
✅ Fine-tuning completed

🚀 STAGE 4: Deploying new model...
✅ Model saved to /home/shiv2077/dev/RETINA/output/bgad_retrained_temp.pt
✅ Backed up existing model to /home/shiv2077/dev/RETINA/output/bgad_backup_20240115_021523.pt
✅ Deployed new model to /home/shiv2077/dev/RETINA/output/bgad_production.pt
✅ Verified production model (45.3 MB)

✅ STAGE 5: Updating metadata...
Marking 12 annotations as processed...
✅ Annotations marked as processed

================================================================================
✅ NIGHTLY RETRAIN COMPLETE
================================================================================
Duration: 3.2 minutes
New anomalies trained: 12
Model deployed: True
Logfile: /home/shiv2077/dev/RETINA/logs/retrain_20240115_021523.log
================================================================================
```

## Troubleshooting

### Issue: "No such file or directory: scripts/nightly_retrain.py"

**Solution:** Verify the script exists and path is correct:
```bash
ls -la /home/shiv2077/dev/RETINA/scripts/nightly_retrain.py
```

### Issue: "No base dataset found"

**Solution:** The script looks for datasets in these locations (in order):
1. `/home/shiv2077/dev/RETINA/decospan_small/train/good/`
2. `/home/shiv2077/dev/RETINA/data/train/good/`
3. `/home/shiv2077/dev/RETINA/MSVPD_Unified_Dataset/train/good/`
4. `/home/shiv2077/dev/RETINA/mvtec_anomaly_detection/bottle/train/good/`

Ensure at least ONE of these exists with normal training images.

### Issue: "Only N new annotations. Minimum required: 5. Skipping retrain."

**Solution:** This is expected behavior - retrain only happens when there are enough new labeled anomalies. This prevents retraining on insufficient data.

- Keep annotating through the active learning queue
- Once you have 5+ new anomalies, the next cron job will retrain

### Issue: Cron job not executing

**Check:**
1. Is cronservice running?
   ```bash
   sudo systemctl status cron
   ```

2. Are environment variables available?
   - Cron runs with minimal environment
   - Add full path to Python (not just `python3`)

3. Working directory exists?
   ```bash
   ls -la /home/shiv2077/dev/RETINA
   ```

### Issue: ImportError for custom modules

**Solution:** If cron can't find `src.backend.models.bgad`, the working directory change may not be working. Try absolute path:

```cron
0 2 * * * /usr/bin/python3 /home/shiv2077/dev/RETINA/scripts/nightly_retrain.py >> /home/shiv2077/dev/RETINA/logs/retrain.log 2>&1
```

## Production Best Practices

### 1. Email Notifications on Failure

Add to cron (sends email if job exits non-zero):

```cron
MAILTO=your-email@domain.com
0 2 * * * cd /home/shiv2077/dev/RETINA && /usr/bin/python3 scripts/nightly_retrain.py >> logs/retrain.log 2>&1
```

### 2. Model Backup Strategy

The script automatically:
- ✅ Creates backup before deploying new model
- ✅ Restores backup if deployment fails
- ✅ Keeps backups in `output/bgad_backup_*.pt`

Review backups:
```bash
ls -lh /home/shiv2077/dev/RETINA/output/bgad_backup_*.pt
```

### 3. Log Rotation

After running for months, logs directory may grow large. Set up log rotation:

```bash
# Create log rotation config
sudo nano /etc/logrotate.d/retina-retrain
```

Add:
```
/home/shiv2077/dev/RETINA/logs/retrain*.log {
    daily
    rotate 30
    compress
    missingok
    notifempty
}
```

Then:
```bash
sudo logrotate /etc/logrotate.d/retina-retrain
```

### 4. Monitor Production Model Updates

Check when model was last updated:

```bash
ls -la /home/shiv2077/dev/RETINA/output/bgad_production.pt
```

### 5. Manual Retrain (If Needed)

You can still manually trigger:

```bash
cd /home/shiv2077/dev/RETINA
python scripts/nightly_retrain.py
```

## Exit Codes

The script returns:
- `0` = Success
- `1` = Skipped (not enough new annotations)
- `2` = Failed with error

Check exit code in cron logs:
```bash
tail /var/log/syslog | grep nightly_retrain
```

## Next Steps

1. ✅ Run script manually to verify it works
2. ✅ Set up cron job
3. ✅ Monitor logs the first few nights
4. ✅ Verify model is being deployed
5. ✅ Check that annotations are being marked as processed

## Summary

The nightly retrain pipeline now:
1. **Automatically discovers** new annotated anomalies from cascade queue
2. **Combines them** with normal training data (25:1 ratio)
3. **Fine-tunes** BGAD model with appropriate learning rate
4. **Deploys** updated weights to production
5. **Backs up** previous model automatically
6. **Logs everything** for monitoring

The complete active learning loop is now closed! 🎉
