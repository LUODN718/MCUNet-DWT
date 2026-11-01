# DWT full-run interruption

## Training state

- `run/checkpoints/checkpoints.json` and the latest checkpoint SHA256 are valid.
- Latest committed state is step 655, sampler position 20,960 in epoch 1. `run/progress.json` is stale at step 600; restart must use the verified checkpoint, not that progress file.
- At inspection, no training worker or controller was running. `controller.json` still says `running` because its writer exited before publishing a terminal state.
- The run log shows committed progress only through step 600, followed by a resource pause.

## Host events

- Windows System log event 1074 at 2026-10-31 17:18:29 records `StartMenuExperienceHost.exe` initiating a restart. The reboot at 17:19:05 terminated the DWT worker and controller. This accounts for the stale `running` record.
- NVIDIA `NvContainerLocalSystem` repeatedly exits with service code 14109. `NvContainerLocalSystem.log` identifies mandatory `NvTelemetryPlugin` initialization failure.
- Windows Application Error events report `rundll32.exe` crashing in `C:\Program Files\NVIDIA Corporation\NvStreamSrv\rxdiag.dll`, exception `0xc0000005` (access violation). Hundreds of `RxDiag Message Pump` processes accumulated after reboot.
- The service recovery configuration repeatedly retries the failing NVIDIA service and runs `C:\Windows\NvContainerRecovery.bat`. A temporary no-restart policy is applied while this training run is resumed; the original recovery actions are recorded and restorable in `host_recovery/`.

These NVIDIA events explain process churn and unstable available memory. The evidence does not attribute the Windows restart to the training process; the recorded initiator is StartMenuExperienceHost.

## Recovery

Resume the exact existing run from checkpoint step 655 with the frozen protocol. Do not reset the model, sampler, optimizer, scheduler, BN state or RNG. Monitor controller and checkpoint metadata rather than relying only on `progress.json`.

Recovery was applied without deleting training data or checkpoints: the noisy NVIDIA service recovery actions were temporarily cleared, the 584 currently matched RxDiag processes were terminated under the user's prior authorization, and attempt 001 resumed from the step-655 checkpoint. It had committed step 700 at the first follow-up check. All logs, reports and recovery scripts are under this `fyp` directory.

Restore the original NVIDIA failure actions after the training run ends with `host_recovery/restore_nvcontainer_recovery.ps1`. The GPU compute driver itself remains enabled.

## 2026-10-31 memory pause and resume

- The resumed full-run worker reached step 1,530, then its resource guard paused because `GlobalMemoryStatusEx` available RAM fell below 2 GiB. The committed checkpoint was preserved; no model, optimizer, scheduler, BN, sampler or RNG state was reset.
- At inspection, the 16 GiB host had a 1.3 GiB training-process working set, while `AWCC.Service` had grown to about 1.5 GiB. The nonpaged pool was about 1.2 GiB. Restarting AWCC reduced its working set; the service was restored after training resumed.
- Steam, WeGame and Edge were closed. Dell TechHub and SupportAssist services were temporarily stopped; AWCC remains running. Their startup configuration was not changed.
- The guard requires over 4 GiB available for 120 seconds before resuming. After memory remained above that watermark, the worker resumed from its checkpoint at step 1,531 and later committed step 1,800. Training logs and resource samples remain under `run_logs/` and `run/resources/`.
- This was a host-resource intervention only; the training recipe and run contract were unchanged. AWCC memory use and `run/resources/latest.json` should be checked if another pause occurs.

## 2026-10-31 recurrent pause and sustained-run mitigation

- The worker resumed and reached step 3,004, then paused again after host available RAM crossed below 2 GiB. The resource policy latches pause until available RAM exceeds 4 GiB for 120 seconds; intermediate recovery to 2-4 GiB does not resume it.
- AWCC and its OCControl child had again grown to about 0.7 GiB each. To prevent recurrence during this run, `AWCCService` is stopped until training completes; its startup type remains Automatic. Firmware/embedded-controller thermal management remains active. GPU temperature was 44-49 C during the pause and restart.
- Dell TechHub and SupportAssist services remain stopped with their Automatic startup settings unchanged. Windows Search was temporarily disabled to establish headroom, then restored to Automatic and Running once the worker resumed. OneDrive, the network proxy, Defender and Codex were kept running.
- The worker resumed from the existing checkpoint and committed step 3,200. Current resource action was `continue`, host available RAM about 3.7 GiB, with training recipe/config/checkpoint state intact.
- After the 20-epoch run completes, restart `AWCCService`, Dell TechHub and SupportAssist. Update or repair AWCC before the next long run because its service memory growth was the recurring trigger.

## 2026-10-31 final auto-restart fix

- Process termination alone was insufficient: DellClientManagement and DellConnectedServiceDelivery restarted Dell TechHub and SupportAssist, which consumed about 1 GiB again. The recurring low-water pause was therefore caused by service auto-restart, not by the DWT step itself.
- For the remainder of this run, `AWCCService`, `DellClientManagementService`, `DellConnectedServiceDelivery`, `DellTechHub`, `SupportAssistAgent` and `Dell SupportAssist Remediation` are stopped with their startup type temporarily set to Disabled. The original startup type was Automatic; restore all six after training with `host_recovery/restore_training_services.ps1`.
- Windows Search and OneDrive have been restored. The training recipe, checkpoint state and frozen resource thresholds remain unchanged.
- The worker resumed, completed epoch 1 development evaluation, and reached step 4,100 in epoch 2. Epoch 1 dev CE was 1.815227 and dev Top-1 was 56.835%. The latest resource action is `continue` at about 3.65 GiB host available RAM; AWCC and Dell processes remain stopped, so they cannot recreate the earlier pressure during this run.

## 2026-10-31 post-resume status and host process audit

- At 21:12 local time, both controller PID 35040 and worker PID 15348 were alive. The worker's progress advanced from step 7,000 to 7,100 over a 20-second verification interval; `controller.json`, `progress.json` and the latest checkpoint agreed on step 7,100 in epoch 2. The resource guard action was `continue`, so this was not a current training interruption.
- Available physical RAM was about 3.0 GiB. The largest resident processes were the training worker (~554 MiB), Defender (~626 MiB), and Codex/ChatGPT processes (~1.2 GiB combined). FlClash and its core used about 229 MiB; OneDrive about 182 MiB; Phone Link about 168 MiB. No unrelated single process was using multiple GiB.
- Windows memory counters showed a ~1.18 GiB nonpaged pool and ~4.78 GiB system cache. The worker's private commit (~3.4 GiB) is not the same as its resident working set (~0.55 GiB). This snapshot does not justify killing system security, network proxy, sync, or Codex processes; the previously identified AWCC/Dell auto-restarting services remain stopped/disabled for this run.

## 2026-11-01 resident file-cache pressure mitigation

- At about 01:17, the worker was genuinely resource-paused. Both processes were alive, but available physical RAM was only about 1.7 GiB. The pause checkpoint was step 32467; the less frequently written progress file still showed 32400. SHA256 of `4b5db957953cb9e85bf03458f1ec4c6be57644f95f3020689d89ed98f051b676.pt` matched its content-addressed filename. There was no training traceback.
- All six previously disabled AWCC/Dell services remained stopped. No unrelated ordinary process used multiple GiB. Further measurement separated the ~5.1 GiB resident system file cache from only ~0.6 GiB standby cache; standby was already part of available RAM. A non-elevated standby-purge attempt returned NTSTATUS 0xC0000061. A bounded allocation/release attempt failed to recover sufficient headroom and was abandoned; it is not an ongoing mitigation.
- At 01:35:59, after the user accepted Windows UAC, helper PID 24724 enabled SeIncreaseQuotaPrivilege and applied the documented SetSystemFileCacheSize API with a 2 GiB hard maximum and no hard minimum. Original minimum=1048576, maximum=17592186044416 and flags=0 were saved before mutation. Available RAM immediately increased from ~2.25 GiB to ~5.35 GiB; the resident-system-cache counter then measured 2 GiB.
- The controller and training worker were not stopped or reset. The original 2 GiB pause / 4 GiB resume / 120-second healthy-period policy stays intact. Model, optimizer, scheduler, sampler, BN, RNG, dataset, batch and scientific source files were untouched. `run/config.json` SHA256 remained `81ee524370082388775021ccbbcb8d8e4cd4f721973379a77b927dff7455c512`.
- The cache helper is bound to the exact controller process through an open process handle and automatically restores original native cache settings on controller exit or run completion. Manual recovery is `host_recovery/restore_cache_limit.ps1` with administrator privileges.
- The Windows power plan also had AC idle sleep set to 600 seconds. At 01:37:10, the separate lightweight `training_awake.py` helper requested system execution availability for this controller's lifetime; no persistent power-plan setting changed. Its combined status file distinguishes resource pause from a live running controller and records checkpoint step separately from the progress file.
- At 01:38:09 the unchanged resource guard released its latch after the full healthy-resource interval. The same worker continued with step 32468 and subsequently saved checkpoints at 32500, 32700 and 32900. At 01:40:24 (epoch 9), the guard still reported `continue`, available RAM was about 5.2 GiB, the resident cache measured ~2 GiB, and both training/helper stderr files were empty. There was no controller/worker restart, batch-size change or source-contract migration. `host_recovery/recovery_validation.json` records the source hashes, native cache settings, memory samples and verified checkpoint hash.
