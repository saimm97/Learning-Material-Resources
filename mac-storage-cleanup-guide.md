# MacBook Storage Cleanup Guide

A step-by-step guide for freeing up disk space, based on a full cleanup session.
Run sections in order. Skip anything you don't recognize using, and read the
⚠️ warnings before running destructive commands.

---

## 1. Check Current Disk Usage

```bash
df -h /
df -h /System/Volumes/Data
diskutil apfs list
```

`diskutil apfs list` gives the real picture — macOS splits storage across
System/Data/VM/Preboot volumes that all share one pool, so a single `df -h /`
line can be misleading. Look at **"Data" volume → Capacity Consumed** for the
real number.

---

## 2. Find What's Using Space

Run these one at a time (not all at once — some can take a few minutes,
especially on iCloud-synced folders like Desktop/Documents):

```bash
du -sh ~/Downloads
du -sh ~/Documents
du -sh ~/Movies
du -sh ~/Desktop
du -sh ~/Library
```

Break down Library further if it's large:

```bash
du -sh ~/Library/Application\ Support/*/ 2>/dev/null | sort -rh | head -20
du -sh ~/Library/Caches/*/ 2>/dev/null | sort -rh | head -20
du -sh ~/Library/Developer/*/ 2>/dev/null | sort -rh | head -10
du -sh ~/Library/Containers/*/ 2>/dev/null | sort -rh | head -15
du -sh ~/Library/Group\ Containers/*/ 2>/dev/null | sort -rh | head -10
```

Find any individual large files/folders anywhere in your home directory
(most effective single command — run this early next time):

```bash
find ~ -xdev -size +1G -exec du -sh {} \; 2>/dev/null | sort -rh | head -40
```

Rule out another user account hogging space:

```bash
du -sh /Users/*/ 2>/dev/null
```

Check for Time Machine local snapshots:

```bash
tmutil listlocalsnapshots /
```

---

## 3. Safe Cache Cleanup (App Caches)

These regenerate automatically — safe to delete, apps may load slightly
slower the first time after.

**Quit the apps first (Chrome, Slack, Cursor, VS Code, Claude) before running these.**

```bash
# General user caches
rm -rf ~/Library/Caches/*
rm -rf ~/Library/Logs/*
rm -rf ~/.Trash/*

# Chrome (does NOT log you out — Cookies/Login Data are separate files, untouched)
rm -rf ~/Library/Application\ Support/Google/Chrome/Default/Cache
rm -rf ~/Library/Application\ Support/Google/Chrome/Default/Code\ Cache
rm -rf ~/Library/Application\ Support/Google/Chrome/Default/Service\ Worker/CacheStorage

# Claude desktop app
rm -rf ~/Library/Application\ Support/Claude/Cache
rm -rf ~/Library/Application\ Support/Claude/Code\ Cache
rm -rf ~/Library/Application\ Support/Claude/GPUCache

# Cursor / VS Code
rm -rf ~/Library/Application\ Support/Cursor/Cache
rm -rf ~/Library/Application\ Support/Cursor/CachedData
rm -rf ~/Library/Application\ Support/Code/Cache
rm -rf ~/Library/Application\ Support/Code/CachedData

# Slack
rm -rf ~/Library/Application\ Support/Slack/Cache
rm -rf ~/Library/Application\ Support/Slack/Service\ Worker/CacheStorage
```

⚠️ Leave `~/Library/Application Support/Claude/vm_bundles` alone — this is a
downloaded VM component (used for Claude's code execution feature), not
cache. Deleting it just forces a large re-download.

---

## 4. Dev Tool & Package Manager Caches

All safe — these are regenerated automatically by their respective tools.

```bash
rm -rf ~/Library/Caches/Yarn/*
rm -rf ~/Library/Caches/Google/*
rm -rf ~/Library/Caches/pip/*
rm -rf ~/Library/Caches/typescript/*
rm -rf ~/Library/Caches/ms-playwright/*
rm -rf ~/Library/Caches/ms-playwright-go/*
rm -rf ~/Library/Caches/node-gyp/*
rm -rf ~/Library/Caches/composer/*
rm -rf ~/Library/Caches/go-build/*
rm -rf ~/Library/Caches/Homebrew/*
rm -rf ~/Library/Caches/virtualenv/*
rm -rf ~/Library/Caches/next-swc/*
rm -rf ~/Library/Caches/JetBrains/*

# Homebrew package cache
brew cleanup -s
rm -rf $(brew --cache)

# npm / yarn / pip (alternate method)
npm cache clean --force
yarn cache clean
pip cache purge
```

---

## 5. Xcode & iOS Simulator Cleanup

Very common source of 30GB+ on dev machines. Simulator devices are full disk
images that pile up across Xcode versions.

```bash
# Xcode build cache — 100% safe, rebuilds automatically
rm -rf ~/Library/Developer/Xcode/DerivedData/*

# Remove old/orphaned simulator devices tied to removed Xcode/iOS versions
xcrun simctl delete unavailable

# List current simulators
xcrun simctl list devices

# ⚠️ Nuclear option — deletes ALL simulator devices and their data.
# Only run if you don't need any current simulator test data.
rm -rf ~/Library/Developer/CoreSimulator/Devices/*
```

---

## 6. Docker Cleanup

⚠️ **Check `docker ps -a` and `docker system df` first if you have active
projects using Docker** — some containers (e.g. databases) have volumes with
real data you don't want to lose.

```bash
# See what's running/stopped
docker ps -a

# See space breakdown: Images / Containers / Volumes / Build Cache
docker system df
```

Safe, non-destructive cleanup (does not touch volumes or tagged images):

```bash
docker builder prune -a -f      # clears build cache
docker image prune -f           # removes only untagged/dangling images
```

More aggressive (only if you're OK re-pulling/rebuilding images later —
does NOT touch volumes/data by itself):

```bash
docker image prune -a -f
```

⚠️ **Do NOT run** `docker system prune -a --volumes` if you have containers
with database data (Postgres, MySQL, Redis with persistence, etc.) that
aren't backed up elsewhere — this deletes volumes permanently.

Check Docker's actual disk usage on macOS:

```bash
du -sh ~/Library/Containers/com.docker.docker
```

---

## 7. Android Emulator Cleanup

```bash
# Removes all Android Virtual Devices — safe if you don't actively test
# Android apps. Android Studio will let you recreate emulators later.
rm -rf ~/.android/avd
```

---

## 8. Manual / Case-by-Case Cleanup

Check these individually before deleting — they're not automatic caches:

```bash
# Old installers, large downloads
ls -lhS ~/Downloads | head -20

# Large screen recordings / videos on Desktop
ls -lhS ~/Desktop | head -20

# iPhone backups (can be 20-80GB+ each)
du -sh ~/Library/Application\ Support/MobileSync/Backup 2>/dev/null

# Photos library size (enable "Optimize Mac Storage" in Photos > Settings > iCloud if large)
du -sh ~/Pictures/Photos\ Library.photoslibrary 2>/dev/null

# iCloud Drive local trash (empty manually via Finder > iCloud Drive > right-click Trash > Empty)
```

---

## 9. Verify Results

```bash
df -h /
df -h /System/Volumes/Data
du -sh ~/Library/Application\ Support
du -sh ~/Library/Developer
du -sh ~/Library/Caches
```

---

## Quick Reference: What's Always Safe to Delete

| Path | Safe? | Notes |
|---|---|---|
| `~/Library/Caches/*` | ✅ | Regenerates automatically |
| `~/Library/Logs/*` | ✅ | No functional impact |
| `~/.Trash/*` | ✅ | Same as emptying Trash in Finder |
| `~/Library/Developer/Xcode/DerivedData` | ✅ | Xcode rebuild cache |
| `~/Library/Developer/CoreSimulator/Devices` | ✅ | Simulator devices, will need recreating |
| App-specific `Cache`/`Code Cache`/`GPUCache` folders | ✅ | Browser-style cache |
| `docker builder prune -a -f` | ✅ | Build cache only |
| `~/.android/avd` | ✅ (if unused) | Android emulator data |
| `~/Library/Application Support/Claude/vm_bundles` | ⚠️ | Not cache — functional VM component |
| Docker volumes / `docker system prune --volumes` | ❌ | Can delete real database data |
| Another user's home folder | ❌ | Not yours to delete |

---

## Notes From This Session

- Biggest wins found: iOS Simulator devices (~34GB), Chrome cache (~13GB),
  large screen recordings on Desktop, and stale dev tool caches.
- `du -sh ~/Library` and similar can appear to "freeze" — this is usually
  just slow scanning (especially on iCloud-synced folders), not an actual
  hang. Give it 2-3 minutes before assuming it's stuck; `Ctrl+C` cancels it.
- Prefer Apple menu → About This Mac → Storage → Manage as a faster,
  non-hanging GUI alternative for a quick overview before diving into Terminal.
